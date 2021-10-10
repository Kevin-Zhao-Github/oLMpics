"""
Code for evaluating various models on the oLMpics MLM tasks.

Example usage:
python run_mlm.py t5-large data/number_comparison_age_compare_masked_dev.jsonl 2

Tested models:
"bert-base-uncased"
"distilbert-base-uncased"
"bert-large-uncased"
"bert-large-uncased-whole-word-masking"
"roberta-large"
"facebook/bart-large"
"t5-large"
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
    parser.add_argument("model_name_or_path", help="Huggingface pretrained model name/path")
    parser.add_argument("data_path", help="Path to jsonl data for MLM task")
    parser.add_argument("num_choices", type=int, help="Number of answer choices for task")
    parser.add_argument(
        "--max_seq_length",
        default=25,
        type=int,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--seed",
        default=123,
        type=int,
    )
    parser.add_argument(
        "--sample_eval",
        default=-1,
        type=int,
        help="Number of examples to evaluate on, default of -1 evaluates on all examples"
    )
    parser.add_argument(
        "--num_heads_disabled",
        default=0,
        type=int,
        help="Number of heads to disable"
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
        out = tokenizer(questions, max_length=max_length, padding="max_length")
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
        out = tokenizer(questions, max_length=max_length, padding="max_length")
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


def evaluate(args, model, tokenizer, eval_dataset, output_confidence=False):
    """
    Args:
        args:
            hyperparameters set using get_args()
        model:
            Huggingface model which will be used for evaluation
        tokenizer:
            Huggingface tokenizer
        output_confidence:
            If True, will output the naive "confidence" measure of attention heads
            See "Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned" (Voita) for details

    Returns: Tuple of answers, preds, (confidence)
        answers - list of ground-truth labels
        preds - list of labels predicted by model
        confidence - included if output_confidence=True, tensor with confidence ratings
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
        LABELS = tokenizer("<extra_id_0>", add_special_tokens=False, return_tensors="pt")
        LABELS = LABELS.input_ids.to(args.device)

    all_answers = []
    all_preds = []

    if output_confidence:
        all_attentions = torch.zeros((model.config.num_hidden_layers, model.config.num_attention_heads))

    head_mask = torch.ones(model.config.num_hidden_layers, model.config.num_attention_heads)
    if args.num_heads_disabled > 0:
        assert any(prefix in args.model_name_or_path.lower() for prefix in ("roberta", "bert")), "Not tested for other models"
        assert args.num_heads_disabled < model.config.num_hidden_layers * model.config.num_attention_heads, \
            f"Model only has {model.config.num_hidden_layers * model.config.num_attention_heads} heads"
        random_pairs = []
        while len(random_pairs) < args.num_heads_disabled:
            random_pair = (random.randint(0, model.config.num_attention_heads-1), random.randint(0, model.config.num_hidden_layers-1))
            if random_pair not in random_pairs:
                random_pairs.append(random_pair)

        for pair in random_pairs:
            head_mask[pair[1], pair[0]] = 0

    for batch in eval_dataloader:
        model.eval()

        # batch["choice_list"] is [num_choices, batch_size]
        for i in range(len(batch["choice_list"][0])):
            all_answers.append(batch["choice_list"][batch["answer_id"][i]][i])

        choice_lists = batch.pop("choice_list")
        batch_len = len(batch["answer_id"])
        del batch["answer_id"]
        for key in batch:
            batch[key] = torch.stack(batch[key], dim=-1).to(args.device)

        with torch.no_grad():
            if "t5" not in args.model_name_or_path.lower():
                outputs = model(**batch, output_attentions=True, head_mask=head_mask)
            else:
                outputs = model(input_ids=batch["input_ids"], decoder_input_ids=batch["input_ids"])
#                 BATCH_LABELS = LABELS.repeat(batch_len, 1)
#                 outputs = model(input_ids=batch["input_ids"], labels=BATCH_LABELS)

            if output_confidence:
                attentions = torch.stack(outputs.attentions) #[:,:,:,:-1, :-1]

                for b in range(attentions.size()[1]):
                    #sep_ind = (batch["input_ids"][b] == tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)[0]).nonzero(as_tuple=True)[0].item()
                    sep_ind = (batch["input_ids"][b] == tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)[0]).nonzero(as_tuple=True)[0].item()
                    for seq_ind1 in range(attentions.size()[-1]):
                        for seq_ind2 in range(attentions.size()[-1]):
                            if seq_ind1 == sep_ind or seq_ind2 == sep_ind or seq_ind1 == 0 or seq_ind2 == 0:
                                attentions[:, b, :, seq_ind1, seq_ind2] = 0

                maxes = torch.amax(attentions, dim=(3, 4))
                sums = torch.sum(maxes, dim=1)
                torch.add(all_attentions, sums, out=all_attentions)

            logits = outputs.logits
            choice_ids = []

            for i, logit in enumerate(logits):  # Assuming all are single tokens
                choice_ids = torch.tensor([tokenizer.encode(" " + choice_lists[j][i], add_special_tokens=False)[0] for j in range(len(choice_lists))])
                if "t5" in args.model_name_or_path.lower():

                    probs = logit[0].index_select(0, choice_ids).to(args.device)
                else:
                    MASK_INDEX = batch["input_ids"][i].tolist().index(MASK_ID)
                    probs = logit[MASK_INDEX].index_select(0, choice_ids).to(args.device)

                max_ind = torch.argmax(probs)
                all_preds.append(choice_lists[max_ind][i])

    output = (all_answers, all_preds)
    if output_confidence:
        torch.div(all_attentions, len(eval_dataloader) * args.per_device_eval_batch_size, out=all_attentions)
        output += (all_attentions,)

    print(len(output))
    print(output[-1])
    return output


def main():
    args = get_args()
    transformers.set_seed(args.seed)

    logger.info("Loading model.")
    if "t5" in args.model_name_or_path.lower():
        model = transformers.T5ForConditionalGeneration.from_pretrained(args.model_name_or_path).to(args.device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.mask_token = "<extra_id_0>"
    elif "gpt" in args.model_name_or_path.lower():
        raise NotImplementedError
        # model = transformers.AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)#.cuda()
        # args.per_device_eval_batch_size = 1
        # tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
        # tokenizer.mask_token = "[MASK]"
    else:
        model = transformers.AutoModelForMaskedLM.from_pretrained(args.model_name_or_path).to(args.device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)


    eval_questions, eval_choices, eval_answer_ids = get_data(args.data_path, args.sample_eval, args.num_choices)
    AgeDataset = RoBERTaDataset if any(prefix in args.model_name_or_path.lower() for prefix in ("roberta", "bart", "distil", "electra", "t5")) else BERTDataset
    eval_dataset = AgeDataset(eval_questions, eval_choices, eval_answer_ids, tokenizer, args.max_seq_length)

    all_answers, all_preds = evaluate(args, model, tokenizer, eval_dataset, output_confidence=True)[:2]  # tuple: answers, preds, (confidence)

    logger.info(f"Accuracy: {(np.array(all_answers) == np.array(all_preds)).mean()}")


if __name__ == "__main__":
    main()
