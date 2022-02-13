#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pretraining the library models for T5-like span-masked language modeling on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be pretrained by this script:
https://huggingface.co/models?filter=t5
"""
import logging
import os
import sys
from dataclasses import asdict, dataclass, field

# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
from enum import Enum
from itertools import chain
import pickle
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm
import wandb

import torch
import torch.nn.functional as F

from datasets import load_dataset
import transformers
from accelerate import Accelerator

from t5_utils import NoamLR, Adafactor


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of steps before backprop."}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Ex: bart"},
    )
    model_resume_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path of model checkpoint, if resuming training"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_pickle_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pickle file with processed data, using this argument means datasets library is not used."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word masking in Chinese."},
    )
    validation_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input validation ref data file for whole word masking in Chinese."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization and masking. Sequences longer than this will be truncated. Default to the max input length of the model."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for span masked language modeling loss"}
    )
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean span length of masked tokens"},
    )

    def __post_init__(self):
        if self.dataset_pickle_path is None and self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file or path to processed pickled data.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


@dataclass
class DataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: transformers.PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int

    # def __call__(self, examples: List[Dict[str, torch.tensor]]) -> Dict[str, torch.tensor]:
    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, torch.tensor]:

        # convert list to dict and tensorize input
        batch = transformers.BatchEncoding(
            {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )

        input_ids = batch["input_ids"]
        batch_size, expanded_input_length = input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(expanded_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = torch.tensor(self.filter_input_ids(input_ids, input_ids_sentinel))
        batch["labels"] = torch.tensor(self.filter_input_ids(input_ids, labels_sentinel))

        if batch["input_ids"].size()[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but should be {self.target_length}."
            )

        if batch["labels"].size()[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be {self.target_length}."
            )

        # to check that tokens are correctly proprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices
        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]
        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        input_ids = input_ids_full[input_ids_full > 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


def norms_to_tensor(encoder_norms):
    norm_list = []
    for el in encoder_norms:
        norm_list.append(el[0])  # .detach().cpu())  # TODO: was el[1]

    # del encoder_norms
    return torch.stack(norm_list, dim=1)


def compute_specialization_metric(encoder_norms, device):
    """
    Args:
        encoder_norms - Attention norms.
                        Tensor of size (batch_size, num_layers, num_heads, seq_len, seq_len)
    """

    batch_size, num_layers, num_heads, seq_len, seq_len2 = encoder_norms.size()
    assert seq_len == seq_len2
    encoder_norms = encoder_norms.permute(0, 2, 1, 3, 4)  # flip layer dimension with head dimension
    encoder_norms = F.normalize(encoder_norms.flatten(3).contiguous(), p=1.0,
                                dim=-1)  # divide each attention norm pattern by its mean
    # encoder_norms now has a size of (num_heads, num_layers, seq_len * seq_len)

    metric_list = []
    for encoder_norm in encoder_norms:
        head_means = []
        for single_head_norms in encoder_norm:
            single_head_distances = F.pdist(single_head_norms, p=1)  # pairwise L1 distances
            head_means.append(torch.mean(single_head_distances))

        single_head_mean = torch.stack(head_means).mean()
        all_head_mean = torch.mean(
            F.pdist(encoder_norm.flatten(start_dim=0, end_dim=1).contiguous(), p=1
                    )).mean()  # pairwise distances between all attention norm patterns

        metric_list.append(single_head_mean.item() / all_head_mean.item())

    return torch.tensor(sum(metric_list), device=device, requires_grad=False), \
           torch.tensor(len(encoder_norms), device=device, requires_grad=False)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    accelerator = Accelerator()

    parser = transformers.HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    if accelerator.is_local_main_process:
       # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            level=logging.INFO,
            datefmt="[%X]",
        )

        logger = logging.getLogger(__name__)

        # Set the verbosity to info of the Transformers logger (on main process only):
        logger.info(f"Training/evaluation parameters {training_args}")

        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
            logger.info(f"Created output_dir at {training_args.output_dir}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    if data_args.dataset_pickle_path is not None:
        if accelerator.is_local_main_process:
            logger.info("Loading processed data from pickle file.")

        with open(data_args.dataset_pickle_path, "rb") as f:
            tokenized_datasets = pickle.load(f)
        if accelerator.is_local_main_process:
            logger.info("Done loading pickle data.")
    else:
        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
        # (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
        # 'text' is found. You can easily tweak this behavior (see below).
        if data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)

            if "validation" not in datasets.keys():
                datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                )
                datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                )
        else:
            data_files = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
            extension = data_args.train_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
            datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

            if "validation" not in datasets.keys():
                datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                )
                datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                )
        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        # Load pretrained model and tokenizer

    if model_args.tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.config_name:
        config = transformers.T5Config.from_pretrained(
            model_args.config_name, cache_dir=model_args.cache_dir, vocab_size=len(tokenizer)
        )

        if model_args.model_type != "t5":
            raise NotImplementedError

        config.decoder_start_token_id = config.pad_token_id
    elif model_args.model_name_or_path:
        config = transformers.T5Config.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = transformers.CONFIG_MAPPING[model_args.model_type]()
        if accelerator.is_local_main_process:
            logger.warning("You are instantiating a new config instance from scratch.")

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    # T5-like span masked language modeling will fuse consecutively masked tokens to a single sentinel token.
    # To ensure that the input length is `max_seq_length`, we need to increase the maximum length
    # according to `mlm_probability` and `mean_noise_span_length`. We can also define the label length accordingly.
    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=max_seq_length,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
    )

    if data_args.dataset_pickle_path is None:
        if training_args.do_train:
            column_names = datasets["train"].column_names
        else:
            column_names = datasets["validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # Since we make sure that all sequences are of the same length, no attention_mask is needed.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_attention_mask=False, truncation=True)

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of expanded_inputs_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= expanded_inputs_length:
                total_length = (total_length // expanded_inputs_length) * expanded_inputs_length
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if accelerator.is_local_main_process:
        wandb.init(project="T5_Pretraining", entity="frostbyte")
        wandb.config.update(training_args)
        wandb.config.update(model_args)
        wandb.config.update(data_args)
        wandb.config.update(config.to_dict())

    # Initialize our training
    if model_args.model_name_or_path:
        model = transformers.T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, config=config, seed=training_args.seed)
    else:
        config.vocab_size = len(tokenizer)
        model = transformers.T5ForConditionalGeneration(config)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    if training_args.adafactor:
        optimizer = Adafactor(optimizer_grouped_parameters, lr=training_args.learning_rate,
                                           scale_parameter=False, relative_step=False)
    else:
        optimizer = transformers.AdamW(
            optimizer_grouped_parameters,
            lr=training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon
        )

    optimizer.zero_grad()

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
        input_length=max_seq_length,
        target_length=targets_length,
        pad_token_id=model.config.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size)
    eval_batch_size = int(training_args.per_device_eval_batch_size)

    train_loader = torch.utils.data.DataLoader(tokenized_datasets["train"], shuffle=True,
                                               collate_fn=data_collator, batch_size=train_batch_size)
    eval_loader = torch.utils.data.DataLoader(tokenized_datasets["validation"], shuffle=False,
                                              collate_fn=data_collator, batch_size=eval_batch_size)

    # num_train_steps = len(tokenized_datasets["train"]) // train_batch_size * num_epochs
    # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, training_args.warmup_steps, num_train_steps)
    scheduler = NoamLR(optimizer, warmup_steps=training_args.warmup_steps)

    if model_args.model_resume_checkpoint is not None:
        if accelerator.is_local_main_process:
            logger.info("Resuming from checkpoint")

        checkpoint = torch.load(model_args.model_resume_checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler = checkpoint["scheduler"]
        resume_step = checkpoint["step"]
    else:
        resume_step = -1

    model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)

    # for epoch in range(num_epochs):
    assert num_epochs == 1
    epoch = 0
    # only the "total" since the last logging step
    total_train_loss = torch.tensor([0.0], device=accelerator.device, requires_grad=False)
    total_train_specialization_metric = torch.tensor([0.0], device=accelerator.device, requires_grad=False)
    total_num_examples = torch.tensor([0.0], device=accelerator.device, requires_grad=False)

    for step, batch in tqdm(enumerate(train_loader), desc="Training", total=len(train_loader),
                            disable=not accelerator.is_local_main_process):
        cur_step = epoch * len(train_loader) + step
        if cur_step <= resume_step:
            continue

        if cur_step % training_args.eval_steps == 0:  # and cur_step > 0:
            if (cur_step) % training_args.gradient_accumulation_steps != 0:
                if accelerator.is_local_main_process:
                    logger.info("Skipping evaluate because gradients are accumulated")

                continue
            eval_loss = torch.tensor([0.0], device=accelerator.device, requires_grad=False)
            eval_specialization_metric = torch.tensor([0.0], device=accelerator.device, requires_grad=False)
            eval_acc = torch.tensor([0.0], device=accelerator.device, requires_grad=False)

            model.eval()
            batch.to("cpu")
            for eval_batch in tqdm(eval_loader, desc="Evaluating", leave=False,
                                   disable=not accelerator.is_local_main_process):
                optimizer.zero_grad()
                loss, decoder_last_state, decoder_cache, decoder_states, decoder_attns, decoder_self_norms, \
                decoder_cross_norms, encoder_last_state, encoder_states, encoder_attns, encoder_norms = \
                    model(**eval_batch, output_hidden_states=True, output_attentions=True, output_norms=True)

                preds = torch.argmax(decoder_last_state, dim=-1).detach().cpu()
                acc = torch.eq(preds, eval_batch["labels"].cpu()).float().sum().to(accelerator.device)
                del preds

                batch_specialization_metric, batch_size = compute_specialization_metric(norms_to_tensor(encoder_norms), accelerator.device)
                del encoder_norms

                eval_loss += loss.detach()
                eval_acc += acc / targets_length
                eval_specialization_metric += batch_specialization_metric
                del batch_specialization_metric, batch_size, loss, acc

            num_eval_examples = len(tokenized_datasets["validation"])
            avg_eval_loss =  accelerator.gather(eval_loss).mean().item() / len(eval_loader)
            avg_eval_specialization_metric = accelerator.gather(eval_specialization_metric).sum().item() / num_eval_examples
            avg_eval_acc = accelerator.gather(eval_acc).sum().item() / num_eval_examples

            if accelerator.is_local_main_process:
                wandb.log({
                    "eval_loss": avg_eval_loss,
                    "eval_specialization_metric": avg_eval_specialization_metric,
                    "eval_acc": avg_eval_acc,
                }, step=cur_step * 2)  # TODO: don't hardcode, multiply by num processes

                del eval_loss, eval_acc, eval_specialization_metric

            batch.to(accelerator.device)

            optimizer.zero_grad()

        model.train()
        loss, decoder_last_state, decoder_cache, decoder_states, decoder_attns, decoder_self_norms, \
            decoder_cross_norms, encoder_last_state, encoder_states, encoder_attns, encoder_norms = \
            model(**batch, output_hidden_states=True, output_attentions=True, output_norms=True)

        accelerator.backward(loss)

        if (cur_step + 1) % training_args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        batch_specialization_metric, batch_size = compute_specialization_metric(norms_to_tensor(encoder_norms), device=accelerator.device)

        total_train_loss += loss.detach()
        total_train_specialization_metric += batch_specialization_metric
        total_num_examples += batch_size

        del loss, batch_specialization_metric, batch_size

        if cur_step % training_args.logging_steps == 0 and cur_step > 0:
            avg_train_loss = accelerator.gather(total_train_loss).mean().item() / training_args.logging_steps
            avg_train_specialization_metric = accelerator.gather(total_train_specialization_metric).mean().item() \
                                              / accelerator.gather(total_num_examples).mean().item()
            if accelerator.is_local_main_process:
                wandb.log({
                    "train_loss": avg_train_loss,
                    "train_specialization_metric": avg_train_specialization_metric,
                    "learning_rate": scheduler.get_last_lr()[0],
                }, step=cur_step * 2)  # TODO: don't hardcode, multiply by num processes

            total_train_loss[0] = 0.0
            total_train_specialization_metric[0] = 0.0
            total_num_examples[0] = 0.0

        if cur_step % training_args.save_steps == 0 and cur_step > 0 and accelerator.is_local_main_process:
            checkpoint = {
                "step": cur_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler
            }
            accelerator.save(checkpoint, f"{training_args.output_dir}/checkpoint_{cur_step // training_args.save_steps}.pt")


if __name__ == "__main__":
    main()
