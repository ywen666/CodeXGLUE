"""
Code completion (both token level and line level) pipeline in CodeXGLUE
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
from datetime import datetime

import torch
import numpy as np
from dataset import JavafinetuneDataset

import transformers
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          GPTNeoConfig, GPTNeoForCausalLM)
from model import RNNModel
from CustomTensorboardCallback import CustomTensorBoardCallback

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'rnn': (GPT2Config, RNNModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'gpt-neo': (GPTNeoConfig, GPTNeoForCausalLM, GPT2Tokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

def update_config(args, config):
    # config.n_positions = config.n_ctx = args.block_size
    config.vocab_size = args.vocab_size

def get_special_tokens(path):
    lits = json.load(open(path))
    tokens = ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"]
    for lit in lits["str"]:
        tokens.append(f"<STR_LIT:{lit}>")
    for lit in lits["num"]:
        tokens.append(f"<NUM_LIT:{lit}>")
    for lit in lits["char"]:
        tokens.append(f"<CHAR_LIT:{lit}>")
    return tokens


def train(args, train_dataset, dev_dataset, model):
    """ Train the model """
    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,

        do_train=True,
        do_eval=args.do_eval,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_gpu_train_batch_size,
        per_device_eval_batch_size=args.per_gpu_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        weight_decay=0.05,
        warmup_steps=args.warmup_steps,
        # max_grad_norm=100000.0,

        logging_dir=args.output_dir,
        logging_first_step=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=10,

        dataloader_drop_last=True,
        dataloader_num_workers=3,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        report_to=args.report_to,
        fp16=args.fp16,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )
    trainer.remove_callback(transformers.integrations.TensorBoardCallback)
    trainer.add_callback(CustomTensorBoardCallback())
    trainer.train()
    if args.local_rank in [0, -1]:
        model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint"))

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_filename", default=None, type=str, required=True,
                        help="The input data path.")
    parser.add_argument("--dev_filename", default=None, type=str, required=True,
                        help="The input data path.")
    parser.add_argument("--langs", default=None, type=str, required=True,
                        help="Languages to train, if all, train all languages in data_dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--model_type", default="gpt2", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--pretrain_dir", default="", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--config_dir", type=str,
                        help="config name. Required when training from scratch")
    parser.add_argument("--tokenizer_dir", type=str,
                        help="Pre-trained tokenizer dir. Required when training from scratch")
    parser.add_argument("--lit_file", type=str,
                        help="literals json file")
    parser.add_argument("--load_name", type=str, default="pretrained",
                        help="Load pretrained model name")

    parser.add_argument("--block_size", default=1024, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--eval_steps", default=1000, type=int,
                        help="")
    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--log_file', type=str, default='')
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--report_to', default='tensorboard', type=str)

    pool = None
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    # Set seed
    set_seed(args)

    # get special tokens
    special_tokens = get_special_tokens(args.lit_file)

    # Load pre-trained model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(
        args.pretrain_dir,
        sep_token='</s>', cls_token='<s>',
        pad_token='<pad>', unk_token='<|UNKNOWN|>',
        additional_special_tokens=special_tokens)
    model = model_class.from_pretrained(args.pretrain_dir)
    model.resize_token_embeddings(len(tokenizer))
    model_parameters = model.parameters()
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Model has a total of {num_params} trainable parameters")
    logger.info("Training/evaluation parameters %s", args)

    # Training
    train_dataset = JavafinetuneDataset(tokenizer, args.train_filename, max_length=args.block_size)
    dev_dataset = JavafinetuneDataset(tokenizer, args.dev_filename, max_length=args.block_size)
    if args.do_train:
        train(args, train_dataset, dev_dataset, model)


if __name__ == "__main__":
    main()