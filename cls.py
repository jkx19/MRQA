from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from model.prefix import BertForQuestionAnswering, BertPrefixModel
from model.prefix import DebertaPrefixModel
from transformers import AutoTokenizer, AutoConfig, DebertaForQuestionAnswering
from transformers.models.bert.configuration_bert import BertConfig
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import set_seed
import torch
from torch.optim import AdamW

from tqdm import tqdm
import argparse
import os
import sys
import json

from data.mrqa_dataset import MRQA

class Train_API():
    
    def __init__(self, args) -> None:
        # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        self.batch_size = args.batch_size*torch.cuda.device_count()
        if args.model == 'bert':
            self.model_name = f'bert-{args.model_size}-uncased'
        elif args.model == 'deberta':
            self.model_name = 'microsoft/deberta-xlarge'

        config = AutoConfig.from_pretrained(
            self.model_name,
            revision='main',
        )
        config.dropout =args.dropout
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision='main',
            use_fast=True,
        )
        config.num_labels = 2
        config.pre_seq_len = args.pre_seq_len
        config.mid_dim = args.mid_dim
        method = args.method
        if args.model == 'bert':
            if method == 'prefix':
                self.model = BertPrefixModel.from_pretrained(
                    self.model_name,
                    config=config,
                    revision='main',
                )
            elif method == 'finetune':
                self.model = BertForQuestionAnswering.from_pretrained(
                    self.model_name,
                    config=config,
                    revision='main',
                )
        elif args.model == 'deberta':
            if method == 'prefix':
                self.model = DebertaPrefixModel.from_pretrained(
                    self.model_name,
                    config=config,
                    revision='main',
                )
            elif method == 'finetune':
                self.model = DebertaForQuestionAnswering.from_pretrained(
                    self.model_name,
                    config=config,
                    revision='main',
                )

        dataset = MRQA(tokenizer, self.batch_size)
        # exit()

        self.eval_example = dataset.eval_example
        self.eval_dataset = dataset.eval_dataset

        self.train_loader = dataset.train_loader
        self.eval_loader = dataset.eval_loader

        self.device = torch.device('cuda:0')
        self.batch_size = self.batch_size * torch.cuda.device_count()
        self.epoch = args.epoch
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.weight_decay = 0
        self.gamma = args.gamma
        self.lr = args.lr
        self.seed = args.seed

        self.compute_metric = dataset.compute_metric
        self.post_process_function = dataset.post_process_function


    def get_optimizer(self):
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]            
        optimizer_kwargs = {
            "betas": (self.adam_beta1, self.adam_beta2),
            "eps": self.adam_epsilon,
        }
        optimizer_kwargs["lr"] = self.lr            
        self.optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

    def get_schedular(self):
        pass

    def train(self):
        self.get_optimizer()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.gamma)
        pbar = tqdm(total=(len(self.train_loader) + len(self.eval_loader))*self.epoch)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        best_dev_result = 0
        best_result = None
        best_model = None
        for epoch in range(self.epoch):
            # Train
            total_loss = 0
            self.model.train()
            for batch_idx, batch in enumerate(self.train_loader):
                batch = {k:v.to(self.device) for k,v in batch.items()}
                output = self.model(**batch)
                loss = torch.sum(output.loss)
                # loss = output.loss
                total_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                pbar.update(1)
            self.scheduler.step()

            result = self.evaluate(pbar)
            eval_f1 = result['eval_f1']
            if eval_f1 > best_dev_result:
                best_dev_result = eval_f1
                best_result = result
                best_model = self.model.prefix_encoder
            pbar.set_description(f'Train_loss: {total_loss:.0f}, Eval_F1: {eval_f1:.2f}')
        torch.save(best_model, f'checkpoints/prefix_{self.seed}_{eval_f1}_{self.model_name}')
        return best_result

    def evaluate(self, pbar: tqdm):
        self.model.eval()
        with torch.no_grad():
            start, end = [],[]
            for batch_idx, batch in enumerate(self.eval_loader):
                batch = {k:v.to(self.device) for k,v in batch.items()}
                output = self.model(**batch)
                start_logits, end_logits = output.start_logits, output.end_logits
                start.append(start_logits)
                end.append(end_logits)
                pbar.update(1)
            start_logits = np.array(torch.cat(start).cpu())
            end_logits = np.array(torch.cat(end).cpu())
        eval_preds = self.post_process_function(self.eval_example, self.eval_dataset, (start_logits, end_logits))
        metrics = self.compute_metric(eval_preds)
        for key in list(metrics.keys()):
                if not key.startswith(f"eval_"):
                    metrics[f"eval_{key}"] = metrics.pop(key)        
        return metrics

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            start, end = [],[]
            for batch_idx, batch in enumerate(self.eval_loader):
                batch = {k:v.to(self.device) for k,v in batch.items()}
                output = self.model(**batch)
                start_logits, end_logits = output.start_logits, output.end_logits
                start.append(start_logits)
                end.append(end_logits)
            start_logits = np.array(torch.cat(start).cpu())
            end_logits = np.array(torch.cat(end).cpu())
        preds = self.post_process_function(self.eval_example, self.eval_dataset, (start_logits, end_logits))
        out_file = open('output/prediction.json', 'w')
        predictions = dict((p["id"], p["prediction_text"]) for p in preds[0])
        json.dump(predictions, out_file)


def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-2)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--pre_seq_len', type=int, default=8)
    parser.add_argument('--mid_dim', type=int, default=512)
    parser.add_argument('--model', type=str, choices=['bert', 'deberta'], default='bert')
    parser.add_argument('--model_size', type=str, choices=['base', 'large'], default='base')
    parser.add_argument('--method', type=str, choices=['finetune', 'prefix'], default='prefix')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--cuda', type=str, default='5')
    parser.add_argument('--seed', type=int, default=44)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = construct_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    set_seed(args.seed)
    train_api = Train_API(args)
    result = train_api.train()
    sys.stdout = open('result.txt', 'a')
    print(args)
    print(result)
