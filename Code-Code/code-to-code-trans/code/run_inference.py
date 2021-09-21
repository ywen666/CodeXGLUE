import argparse
from tqdm.auto import tqdm

import transformers
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          GPTNeoConfig, GPTNeoModel, GPT2Tokenizer)
from run_trainer import JavaClassData

import torch
import torch.nn as nn

from model import Seq2Seq, Seq2SeqPretrained


MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 'gpt-neo': (GPTNeoConfig, GPTNeoModel, GPT2Tokenizer)}

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
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    encoder = model_class.from_pretrained(args.model_name_or_path)
    encoder.resize_token_embeddings(len(tokenizer))
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2SeqPretrained(
        encoder=encoder,decoder=decoder,
        config=config,
        tokenizer=tokenizer
    )
    if args.fp16:
        model.half()

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
        per_device_eval_batch_size=args.eval_batch_size,
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

    eval_dataset = JavaClassData(
        filename=args.test_filename,
        tokenizer=tokenizer)
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset
    )
    eval_dataloader=trainer.get_eval_dataloader(eval_dataset)
    model.config.is_encoder_decoder = True
    model.eval()
    outputs = []
    progress_bar = tqdm(range(len(eval_dataloader)),
                        disable=(args.local_rank not in [-1, 0]))
    for i, inputs in enumerate(tqdm(eval_dataloader)):
        encoder_inputs = inputs['source_ids'].cuda()
        source_mask = inputs['source_mask'].repeat_interleave(
            args.beam_size, dim=0).cuda()
        input_ids = torch.ones((1, 1), device=model.device, dtype=torch.long)
        input_ids = input_ids * tokenizer.cls_token_id
        with torch.no_grad():
            model_kwargs = {
                "encoder_outputs": model.get_encoder()(
                    encoder_inputs.repeat_interleave(args.beam_size, dim=0),
                    attention_mask=source_mask),
                "source_mask": source_mask,
                "decoder_input_ids": input_ids
            }
            output_ids = model.generate(
                input_ids,
                num_beams=args.beam_size,
                early_stopping=True,
                max_length=350,
                **model_kwargs
            )
            text = tokenizer.decode(output_ids[0][1:],
                                    clean_up_tokenization_spaces=False)
            outputs.append(text)
            progress_bar.update(1)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
