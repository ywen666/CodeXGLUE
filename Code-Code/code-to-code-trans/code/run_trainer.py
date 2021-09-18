from __future__ import absolute_import
import os
import sys
import logging
import argparse

import torch
import torch.nn as nn
from model import Seq2Seq
from CustomTensorboardCallback import CustomTensorBoardCallback

import transformers
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          GPTNeoConfig, GPTNeoModel, GPT2Tokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 'gpt-neo': (GPTNeoConfig, GPTNeoModel, GPT2Tokenizer)}

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)


class JavaClassData(torch.utils.data.Dataset):
    def __init__(self, filename, tokenizer, max_source_length=512):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_source_length
        self.examples = read_examples(filename)

    def __len__(self):
        return min(len(self.examples['source']),
                   len(self.examples['target']))

    def __getitem__(self, idx):
        source_tokens = self.tokenizer.tokenize(
            self.examples['source'][idx])[:self.max_source_length-2]
        source_tokens =[self.tokenizer.cls_token] + \
            source_tokens + [self.tokenizer.sep_token]
        source_ids =  self.tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = self.max_source_length - len(source_ids)
        source_ids += [self.tokenizer.pad_token_id]*padding_length
        source_mask += [0]*padding_length

        #if not training:
        #    target_tokens = tokenizer.tokenize("None")
        #else:
        target_tokens = self.tokenizer.tokenize(
            self.examples['target'][idx])[:self.max_target_length-2]

        target_tokens = [self.tokenizer.cls_token] + \
            target_tokens + [self.tokenizer.sep_token]
        target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = self.max_target_length - len(target_ids)
        target_ids += [self.tokenizer.pad_token_id]*padding_length
        target_mask += [0] * padding_length

        model_inputs = {}
        model_inputs['source_ids'] = source_ids
        model_inputs['source_mask'] = source_mask
        model_inputs['target_ids'] = target_ids
        model_inputs['target_mask'] = target_mask
        return model_inputs


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
    parser.add_argument("--partition", default=0, type=int,
                        help="Which partition of data for training.")

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
    parser.add_argument("--eval_steps", default=1000, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--log_steps", default=10, type=int,
                        help="Log steps.")
    parser.add_argument("--save_steps", default=1000, type=int,
                        help="save steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--report_to', default='tensorboard', type=str)
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
    if args.load_model_path is not None:
        state_dict = torch.load(args.load_model_path, map_location="cpu")
        model.load_state_dict(state_dict)

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        do_train=True,
        do_eval=args.do_eval,
        #do_predict=True,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        label_names=['target_ids'],

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
        save_total_limit=50,

        dataloader_drop_last=True,
        #dataloader_num_workers=3,

        local_rank=args.local_rank,

        generation_max_length=args.generation_max_length,
        deepspeed=args.deepspeed,
        report_to=args.report_to,
        fp16=args.fp16,
    )

    train_dataset = JavaClassData(
        filename=args.train_filename,
        tokenizer=tokenizer)
    eval_dataset = JavaClassData(
        filename=args.dev_filename,
        tokenizer=tokenizer)

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.remove_callback(transformers.integrations.TensorBoardCallback)
    trainer.add_callback(CustomTensorBoardCallback())
    trainer.train()

    if args.local_rank == 0:
        torch.save(model.state_dcit(),
                   os.path.join(args.output_dir, "final_checkpoint.pt"))

if __name__ == "__main__":
    main()
