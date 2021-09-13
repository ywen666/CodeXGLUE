from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import trange
from tqdm.auto import tqdm

from bleu import _bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          GPTNeoConfig, GPTNeoModel, GPT2Tokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 'gpt-neo': (GPTNeoConfig, GPTNeoModel, GPT2Tokenizer)}

import transformers
from CustomTensorboardCallback import CustomTensorBoardCallback
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def read_examples(filename):
    """Read examples from filename."""
    examples = {'source': [], 'target': []}
    assert len(filename.split(','))==2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    with open(src_filename) as f1,open(trg_filename) as f2:
        for line1,line2 in zip(f1,f2):
            examples['source'].append(line1.strip()),
            examples['target'].append(line2.strip()),
    #examples['source'] = examples['source'][:700000]
    #examples['target'] = examples['target'][:700000]
    examples['source'] = examples['source'][:600000]
    examples['target'] = examples['target'][:600000]
    return examples


class JavaClassData(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.examples = read_examples(filename)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )
    parser.add_argument("--tokenizer_name", default="",
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files" )
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filenames (source and target files).")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. (source and target files).")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. (source and target files).")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--generation_max_length', type=int, default=512,
                        help="Max length in the generation.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
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
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--log_steps", default=10, type=int,
                        help="Log steps.")
    parser.add_argument("--save_steps", default=500, type=int,
                        help="save steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--deepspeed', default=None, type=str)
    # print arguments
    args = parser.parse_args()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    #tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,do_lower_case=args.do_lower_case)
    #tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        sep_token='<EOL>', cls_token='<s>',
        eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>')

    #budild model
    encoder = model_class.from_pretrained(args.model_name_or_path)
    encoder.resize_token_embeddings(len(tokenizer))
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id,
                  model_type=args.model_type,trainer=True)

    #if args.load_model_path is not None:
    #    logger.info("reload model from {}".format(args.load_model_path))
    #    model.load_state_dict(torch.load(args.load_model_path))

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        do_train=True,
        do_eval=args.do_eval,
        #do_predict=True,
        #evaluation_strategy='steps',
        #eval_steps=args.eval_steps,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        # max_grad_norm=100000.0,

        logging_dir=args.output_dir,
        logging_first_step=True,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=20,

        dataloader_drop_last=True,
        dataloader_num_workers=3,

        local_rank=args.local_rank,

        generation_max_length=args.generation_max_length,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    #train_dataset = JavaClassData(training=True)
    #eval_dataset = JavaClassData(training=False,
    #                             tokenizer=tokenizer,
    #                             args=args)
    #train_dataset = JavaClassData(filename=args.train_filename)
    #eval_dataset = JavaClassData(filename=args.dev_filename)
    from datasets import Dataset
    train_examples = read_examples(args.train_filename)
    eval_examples = read_examples(args.dev_filename)
    train_dataset = Dataset.from_dict(train_examples)
    eval_dataset = Dataset.from_dict(eval_examples)

    def preprocess_function_batched(examples, training=True):
        inputs = [tokenizer.cls_token + ex + tokenizer.cls_token for ex in examples['source']]

        input_encodings = tokenizer(inputs,
                                    max_length=args.max_source_length,
                                    padding='max_length',
                                    truncation=True)
        source_ids = input_encodings['input_ids']
        source_mask = input_encodings['attention_mask']

        targets = [tokenizer.cls_token + ex + tokenizer.sep_token for ex in examples['target']]
        #if training:
        #    targets = [tokenizer.cls_token + ex + tokenizer.sep_token for ex in examples['target']]
        #else:
        #    targets = ['None' for ex in examples]

        target_encodings = tokenizer(targets,
                                     max_length=args.max_source_length,
                                     padding='max_length',
                                     truncation=True)
        target_ids = target_encodings['input_ids']
        target_mask = target_encodings['attention_mask']

        model_inputs = {}
        model_inputs['source_ids'] = source_ids
        model_inputs['source_mask'] = source_mask
        model_inputs['target_ids'] = target_ids
        model_inputs['target_mask'] = target_mask
        return model_inputs

    def preprocess_function(example, training=True):
        source_tokens = tokenizer.tokenize(example['source'])[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length

        #if not training:
        #    target_tokens = tokenizer.tokenize("None")
        #else:
        target_tokens = tokenizer.tokenize(example['target'])[:args.max_target_length-2]

        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length

        model_inputs = {}
        model_inputs['source_ids'] = source_ids
        model_inputs['source_mask'] = source_mask
        model_inputs['target_ids'] = target_ids
        model_inputs['target_mask'] = target_mask
        return model_inputs

    import functools
    #train_preprocess_fn = functools.partial(preprocess_function_batched,
    #                                        training=True)
    #eval_preprocess_fn = functools.partial(preprocess_function_batched,
    #                                       training=False)
    train_preprocess_fn = functools.partial(preprocess_function,
                                            training=True)
    eval_preprocess_fn = functools.partial(preprocess_function,
                                           training=False)

    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            train_preprocess_fn,
            #batched=True,
            num_proc=16,
            desc="Running tokenizer on train dataset",
        )

    with training_args.main_process_first(desc="eval dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            eval_preprocess_fn,
            #batched=True,
            num_proc=8,
            desc="Running tokenizer on validation dataset",
        )

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.remove_callback(transformers.integrations.TensorBoardCallback)
    trainer.add_callback(CustomTensorBoardCallback())
    if args.load_model_path is not None:
        trainer.train(resume_from_checkpoint=args.load_model_path)
    else:
        trainer.train()

    if args.local_rank == 0:
        torch.save(model.state_dict(),
                   os.path.join(args.output_dir, "final_checkpoint"))

if __name__ == "__main__":
    main()