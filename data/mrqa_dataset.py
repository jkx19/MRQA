import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_metric, load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, BertConfig
from transformers import default_data_collator, EvalPrediction
import numpy as np

from utils_qa import postprocess_qa_predictions

class MRQA:

    def __init__(self, tokenizer: AutoTokenizer, batch_size: int) -> None:
        raw_datasets = load_dataset('data/load_dataset.py')
        column_names = raw_datasets['train'].column_names
        self.question_column_name = "question"
        self.context_column_name = "context"
        self.answer_column_name = "answers"

        # print(raw_datasets)
        # exit()

        self.tokenizer = tokenizer

        self.pad_on_right = tokenizer.padding_side == "right" # True
        self.max_seq_len = 384

        train_dataset = raw_datasets['train']
        self.eval_example = raw_datasets['validation']

        train_dataset = train_dataset.map(
            self.prepare_train_dataset,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
        )
        self.eval_dataset = self.eval_example.map(
            self.prepare_eval_dataset,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on validation dataset",
        )

        self.data_collator = default_data_collator

        self.batch_size = batch_size

        self.train_loader = self.get_data_loader(train_dataset, 'train')
        self.eval_loader = self.get_data_loader(self.eval_dataset, 'eval')

        self.metric = load_metric('data/metric.py')

    def prepare_train_dataset(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples['question'] = [q.lstrip() for q in examples['question']]
        
        tokenized = self.tokenizer(
            examples['question' if self.pad_on_right else 'context'],
            examples['context' if self.pad_on_right else 'question'],
            truncation='only_second' if self.pad_on_right else 'only_first',
            max_length=self.max_seq_len,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_maping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")
        tokenized["start_positions"] = []
        tokenized["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized['input_ids'][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            
            sequence_ids = tokenized.sequence_ids(i)
            sample_index = sample_maping[i]
            answers = examples['answers'][sample_index]

            if len(answers['answer_start']) == 0:
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span 
                # (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized["start_positions"].append(cls_index)
                    tokenized["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized["end_positions"].append(token_end_index + 1)
            
        return tokenized

    def prepare_eval_dataset(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples['question'] = [q.lstrip() for q in examples['question']]

        tokenized = self.tokenizer(
            examples['question' if self.pad_on_right else 'context'],
            examples['context' if self.pad_on_right else 'question'],
            truncation='only_second' if self.pad_on_right else 'only_first',
            max_length=self.max_seq_len,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []

        for i in range(len(tokenized["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized["offset_mapping"][i])
            ]
        return tokenized

    def get_data_loader(self, dataset: HFDataset, type:str):
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        if type == 'train':
            sampler = RandomSampler(dataset, generator=generator)
            dataset = dataset.remove_columns([])
        if type == 'eval':
            dataset = dataset.remove_columns(['offset_mapping', 'example_id'])
            sampler = SequentialSampler(dataset)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )

    def compute_metric(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)

    def post_process_function(self, examples, features, predictions, stage='eval'):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=True,
            n_best_size=20,
            max_answer_length=30,
            null_score_diff_threshold=0.0,
            output_dir='output',
            prefix=stage,
        )
        if True: # squad_v2
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex['answers']} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

