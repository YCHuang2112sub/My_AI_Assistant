## play with chatgpt
import os
import sys
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_dataset(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        os.path.dirname(file_path),
        "cached_lm_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )
    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and (
        (not args.reprocess_input_data and not evaluate) or (evaluate and args.use_cached_eval_features)
    ):
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        dataset, tensor_dataset = (
            features_and_dataset["dataset"],
            features_and_dataset["tensor_dataset"],
        )
    else:
        logger.info("Creating features from dataset file at %s", file_path)

        examples = read_examples_from_file(file_path, evaluate, args)

        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            output_mode=args.output_mode,
            # pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            # pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        dataset, tensor_dataset = (
            features["dataset"],
            features["tensor_dataset"],
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"dataset": dataset, "tensor_dataset": tensor_dataset}, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()

    return dataset, tensor_dataset

