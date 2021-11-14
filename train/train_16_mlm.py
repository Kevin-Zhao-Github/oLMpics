"""
Code for training models on the oLMpics MLM tasks.

Example usage:
python train_mlm.py \
bert-base-uncased data/number_comparison_age_compare_masked_train.jsonl \
data/number_comparison_age_compare_masked_dev.jsonl 2

Tested models:
"bert-base-uncased"
"distilbert-base-uncased"
"bert-large-uncased"
"bert-large-uncased-whole-word-masking"
"roberta-large"
"facebook/bart-large"
# "t5-large"
"albert-large-v1"

Download data from: https://github.com/alontalmor/oLMpics/blob/master/README.md
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import transformers
import wandb
from tqdm.auto import tqdm, trange


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_args():
    """ Set hyperparameters """
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name_or_path",
    help="Huggingface pretrained model name/path")

    parser.add_argument("train_data_path",
    help="Path to jsonl training data for MLM task")

    parser.add_argument("eval_data_path",
    help="Path to jsonl development data for MLM task")

    parser.add_argument("num_choices", type=int,
    help="Number of answer choices for task")

    parser.add_argument(
        "--max_seq_length",
        default=25,
        type=int,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--num_train_epochs",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--no_dropout",
        action="store_true"
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.06,
        type=float,
    )
    parser.add_argument(
        "--seed",
        default=123,
        type=int,
    )
    parser.add_argument(
        "--sample_train",
        default=-1,
        type=int,
        help="Number of examples (not batches) to evaluate on, \
        default of -1 evaluates on entire dataset"
    )
    parser.add_argument(
        "--sample_eval",
        default=-1,
        type=int,
        help="Number of examples (not batches) to evaluate on, \
        default of -1 evaluates on entire dataset"
    )
    parser.add_argument(
        "--clip",
        default=1,
        type=float,
        help="Argument for gradient clipping, -1 means no clipping"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()
    return args


def get_data(file_path, sample, num_choices):
    """ Reads data from jsonl file, code taken from original oLMpics authors """
    data_file = open(file_path, "r")
    logger.info("Reading QA instances from jsonl dataset at: %s", file_path)
    item_jsons = []
    item_ids = []
    questions = []
    choice_lists = []
    answer_ids = []
    for line in data_file:
        item_jsons.append(json.loads(line.strip()))

    if sample != -1:
        item_jsons = random.sample(item_jsons, sample)
        logger.info("Sampling %d examples", sample)

    for item_json in tqdm(item_jsons,total=len(item_jsons)):
        item_id = item_json["id"]

        question_text = item_json["question"]["stem"]

        choice_label_to_id = {}
        choice_text_list = []
        choice_context_list = []
        choice_label_list = []
        choice_annotations_list = []

        any_correct = False
        choice_id_correction = 0

        for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
            choice_label = choice_item["label"]
            choice_label_to_id[choice_label] = choice_id - choice_id_correction
            choice_text = choice_item["text"]

            choice_text_list.append(choice_text)
            choice_label_list.append(choice_label)

            if item_json.get('answerKey') == choice_label:
                if any_correct:
                    raise ValueError("More than one correct answer found for {item_json}!")
                any_correct = True


        if not any_correct and 'answerKey' in item_json:
            raise ValueError("No correct answer found for {item_json}!")


        answer_id = choice_label_to_id.get(item_json.get("answerKey"))
        # Pad choices with empty strings if not right number
        if len(choice_text_list) != num_choices:
            choice_text_list = (choice_text_list + num_choices * [''])[:num_choices]
            choice_context_list = (choice_context_list + num_choices * [None])[:num_choices]
            if answer_id is not None and answer_id >= num_choices:
                logging.warning(f"Skipping question with more than {num_choices} answers: {item_json}")
                continue

        item_ids.append(item_id)
        questions.append(question_text)
        choice_lists.append(choice_text_list)
        answer_ids.append(answer_id)

    data_file.close()
    return questions, choice_lists, answer_ids


class BERTDataset(Dataset):
    """ Dataset with token_type_ids (used for BERT, ALBERT) """
    def __init__(self, questions, choices, answer_ids, tokenizer, max_length):
        out = tokenizer(questions, max_length=max_length, padding="max_length", truncation=True)
        self.input_ids = out["input_ids"]
        self.token_type_ids = out["token_type_ids"]
        self.attention_mask = out["attention_mask"]
        self.questions = questions
        self.choices = choices
        self.answer_ids = answer_ids

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "token_type_ids": self.token_type_ids[i],
            "choice_list": self.choices[i],
            "answer_id": self.answer_ids[i],
        }


class RoBERTaDataset(Dataset):
    """ Dataset without token_type_ids (used for RoBERTa, BART, Distil, ELECTRA, T5) """
    def __init__(self, questions, choices, answer_ids, tokenizer, max_length):
        questions = [question.replace('[MASK]', tokenizer.mask_token) for question in questions]
        out = tokenizer(questions, max_length=max_length, padding="max_length", truncation=True)
        self.input_ids = out["input_ids"]
        self.attention_mask = out["attention_mask"]
        self.questions = questions
        self.choices = choices
        self.answer_ids = answer_ids

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "choice_list": self.choices[i],
            "answer_id": self.answer_ids[i],
        }


def evaluate(args, model, tokenizer, eval_dataset, is_train=False):
    """
    Args:
        args:
            hyperparameters set using get_args()
        model:
            Huggingface model which will be used for evaluation
        tokenizer:
            Huggingface tokenizer

    Returns: Tuple of (answers, preds)
        answers - list of ground-truth labels
        preds - list of labels predicted by model
    """
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_device_eval_batch_size)

    logger.info(f"***** Running evaluation  *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating")

    MASK_ID = tokenizer.encode(tokenizer.mask_token, add_special_tokens=False)
    assert len(MASK_ID) == 1
    MASK_ID = MASK_ID[0]
    if "t5" in args.model_name_or_path.lower():
        decoder_ids = tokenizer("<pad> <extra_id_0>", add_special_tokens=False, return_tensors="pt").input_ids.to(args.device)

    all_answers = []
    all_preds = []
    eval_loss = 0

    for batch in eval_dataloader:
        model.eval()

        curr_answers = []
        # batch["choice_list"] is [num_choices, batch_size]
        for i in range(len(batch["choice_list"][0])):
            curr_answers.append(batch["choice_list"][batch["answer_id"][i]][i])

        all_answers.extend(curr_answers)
        choice_lists = batch.pop("choice_list")
        batch_len = len(batch["answer_id"])
        del batch["answer_id"]
        for key in batch:
            batch[key] = torch.stack(batch[key], dim=-1).to(args.device)

        if "t5" not in args.model_name_or_path.lower():
            labels = torch.full((batch["input_ids"].size()[:2]), -100, device=args.device)
            for i, curr_answer in enumerate(curr_answers):
                MASK_INDEX = batch["input_ids"][i].tolist().index(MASK_ID)
                assert len(tokenizer.encode(" " + curr_answer, add_special_tokens=False)) == 1
                labels[i][MASK_INDEX] = tokenizer.encode(" " + curr_answer, add_special_tokens=False)[0]
        else:
            ins_answers = [f"<extra_id_0> {curr_answer} <extra_id_1>" for curr_answer in curr_answers]
            labels = tokenizer(ins_answers, padding='longest', return_tensors="pt").input_ids.to(args.device)

        with torch.no_grad():
            # NEW
            with torch.cuda.amp.autocast():
                assert "t5" in args.model_name_or_path.lower()
                outputs = model(**batch, decoder_input_ids=decoder_ids.repeat(batch_len, 1))

            logits = outputs.logits
            choice_ids = []

            for i, logit in enumerate(logits):  # Assuming all are single tokens
                choice_ids = torch.tensor([tokenizer.encode(" " + choice_lists[j][i], add_special_tokens=False)[0] for j in range(len(choice_lists))])
                if "t5" in args.model_name_or_path.lower():
                    # probs = logit[0].index_select(0, choice_ids.to(args.device))
                    probs = logit[1].index_select(0, choice_ids.to(args.device))
                else:
                    MASK_INDEX = batch["input_ids"][i].tolist().index(MASK_ID)
                    probs = logit[MASK_INDEX].index_select(0, choice_ids.to(args.device))

                max_ind = torch.argmax(probs)
                all_preds.append(choice_lists[max_ind][i])

    return (np.array(all_answers) == np.array(all_preds)).mean()


def train(args, model, tokenizer, train_dataset, eval_dataset):
    eval_acc = evaluate(args, model, tokenizer, eval_dataset)
    logger.info(f"Initial Eval Accuracy: {eval_acc}")
    if args.num_train_epochs == 0:
        wandb.log({"eval_acc": eval_acc})
        return
    train_acc = evaluate(args, model, tokenizer, train_dataset, is_train=True)
    logger.info(f"Initial Train Accuracy: {train_acc}")
    wandb.log({"eval_acc": eval_acc, "train_acc": train_acc})

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

    logger.info(f"***** Running training  *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Batch size = {args.per_device_train_batch_size}")
    train_dataloader = tqdm(train_dataloader, desc="Training", leave=False)

    optimizer = transformers.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps * args.warmup_ratio, len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)

    MASK_ID = tokenizer.encode(tokenizer.mask_token, add_special_tokens=False)
    assert len(MASK_ID) == 1
    MASK_ID = MASK_ID[0]
    accumulated_loss = torch.tensor(0.0).to(args.device)

    for epoch in tqdm(range(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            if args.no_dropout:
                model.eval()
            else:
                model.train()

            # batch["choice_list"] is [num_choices, batch_size]
            curr_answers = []
            for i in range(len(batch["choice_list"][0])):
                curr_answers.append(batch["choice_list"][batch["answer_id"][i]][i])

            choice_lists = batch.pop("choice_list")
            batch_len = len(batch["answer_id"])
            del batch["answer_id"]
            for key in batch:
                batch[key] = torch.stack(batch[key], dim=-1).to(args.device)

            if "t5" not in args.model_name_or_path.lower():
                MASK_INDEX = batch["input_ids"][0].tolist().index(MASK_ID)
                labels = torch.full((batch["input_ids"].size()[:2]), -100, device=args.device)
                # labels = batch["input_ids"].detach().clone()
                for i, curr_answer in enumerate(curr_answers):
                    MASK_INDEX = batch["input_ids"][i].tolist().index(MASK_ID)
                    assert len(tokenizer.encode(" " + curr_answer, add_special_tokens=False)) == 1
                    labels[i][MASK_INDEX] = tokenizer.encode(" " + curr_answer, add_special_tokens=False)[0]
            else:
                labels = []
                for i, curr_answer in enumerate(curr_answers):
                    labels += tokenizer.encode(f"<extra_id_0> {curr_answer} <extra_id_1>", return_tensors="pt")

                labels = torch.stack(labels, dim=0).to(args.device)

            outputs = model(**batch, labels=labels)

            loss = outputs.loss
            accumulated_loss += float(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                wandb.log({"train_loss": accumulated_loss.item()})
                wandb.log({"lr": scheduler.get_last_lr()[0]})
                loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                accumulated_loss = torch.tensor(0.0).to(args.device)

        eval_acc = evaluate(args, model, tokenizer, eval_dataset)
        logger.info(f"{epoch}th Eval Accuracy: {eval_acc}")
        train_acc = evaluate(args, model, tokenizer, train_dataset, is_train=True)
        logger.info(f"{epoch}th Train Accuracy: {train_acc}")
        wandb.log({"eval_acc": eval_acc, "train_acc": train_acc})

    return True


def main():
    torch.set_grad_enabled(False)
    args = get_args()
    wandb.init(project="oLMpics", entity="frostbyte", group="google-T5", \
               name=f"{args.model_name_or_path}_{args.train_data_path[5:-6]}")
    wandb.config.update(args)

    transformers.set_seed(args.seed)

    logger.info("Loading model.")
    if "t5" in args.model_name_or_path.lower():
        # with torch.cuda.amp.autocast():
        # model = transformers.T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        # model = torch.cuda.amp.initialize(model, opt_level="O1").to(args.device)
        model = transformers.T5ForConditionalGeneration.from_pretrained(args.model_name_or_path).half().to(args.device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.mask_token = "<extra_id_0>"
    else:
        raise NotImplementedError

    # model.gradient_checkpointing_enable()
    wandb.watch(model, log_freq=50)
    train_questions, train_choices, train_answer_ids = get_data(args.train_data_path, args.sample_train, args.num_choices)
    eval_questions, eval_choices, eval_answer_ids = get_data(args.eval_data_path, args.sample_eval, args.num_choices)

    AgeDataset = RoBERTaDataset if any(prefix in args.model_name_or_path.lower() \
    for prefix in ("roberta", "bart", "distil", "electra", "t5")) else BERTDataset

    train_dataset = AgeDataset(train_questions, train_choices, train_answer_ids, tokenizer, args.max_seq_length)
    eval_dataset = AgeDataset(eval_questions, eval_choices, eval_answer_ids, tokenizer, args.max_seq_length)
    """
    temp = train_dataset
    train_dataset = eval_dataset
    eval_dataset = temp
    """

    train(args, model, tokenizer, train_dataset, eval_dataset)


if __name__ == "__main__":
    main()
