# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import gc
import shutil
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024):
        if args.local_rank==-1:
            local_rank=0
            world_size=1
        else:
            local_rank=args.local_rank
            world_size=torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_langs_%s"%(args.langs)+"_blocksize_%d"%(block_size)+"_wordsize_%d"%(world_size)+"_rank_%d"%(local_rank))
        if os.path.exists(cached_file) and not args.overwrite_cache:
            if file_type == 'train':
                logger.warning("Loading features from cached file %s", cached_file)
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)

        else:
            self.inputs = []
            if args.langs == 'all':
                langs = os.listdir(args.data_dir)
            else:
                langs = [args.langs]

            data=[]
            for lang in langs:
                datafile = os.path.join(args.data_dir, lang, file_type+'.pkl')
                if file_type == 'train':
                    logger.warning("Creating features from dataset file at %s", datafile)
                # with open(datafile) as f:
                #     data.extend([json.loads(x)['code'] for idx,x in enumerate(f.readlines()) if idx%world_size==local_rank])
                dataset = pickle.load(open(datafile, 'rb'))
                data.extend(['<s> '+' '.join(x['function'].split())+' </s>' for idx,x in enumerate(dataset) if idx%world_size==local_rank])

            # random.shuffle(data)
            data = data
            length = len(data)
            logger.warning("Data size: %d"%(length))
            input_ids = []
            for idx,x in enumerate(data):
                try:
                    input_ids.extend(tokenizer.encode(x))
                except Exception:
                    pass
                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.warning("Rank %d, load %d"%(local_rank, percent))
            del data
            gc.collect()

            length = len(input_ids)
            for i in range(0, length-block_size, block_size):
                self.inputs.append(input_ids[i : i + block_size])            
            del input_ids
            gc.collect()

            if file_type == 'train':
                logger.warning("Rank %d Training %d token, %d samples"%(local_rank, length, len(self.inputs)))
                logger.warning("Saving features into cached file %s", cached_file)
            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])

class finetuneDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024):
        if args.local_rank==-1:
            local_rank=0
            world_size=1
        else:
            local_rank=args.local_rank
            world_size=torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_blocksize_%d"%(block_size)+"_wordsize_%d"%(world_size)+"_rank_%d"%(local_rank))
        if os.path.exists(cached_file) and not args.overwrite_cache:
            if file_type == 'train':
                logger.warning("Loading features from cached file %s", cached_file)
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)

        else:
            self.inputs = []

            datafile = os.path.join(args.data_dir, f"{file_type}.txt")
            if file_type == 'train':
                logger.warning("Creating features from dataset file at %s", datafile)
            with open(datafile) as f:
                data = f.readlines()

            length = len(data)
            logger.info("Data size: %d"%(length))
            input_ids = []
            for idx,x in enumerate(data):
                x = x.strip()
                if x.startswith("<s>") and x.endswith("</s>"):
                    pass
                else:
                    x = "<s> " + x + " </s>"
                try:
                    input_ids.extend(tokenizer.encode(x))
                except Exception:
                    pass
                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.warning("Rank %d, load %d"%(local_rank, percent))
            del data
            gc.collect()

            length = len(input_ids) // world_size
            logger.info(f"tokens: {length*world_size}")
            input_ids = input_ids[local_rank*length: (local_rank+1)*length]

            for i in range(0, length-block_size, block_size):
                self.inputs.append(input_ids[i : i + block_size])            
            del input_ids
            gc.collect()

            if file_type == 'train':
                logger.warning("Rank %d Training %d token, %d samples"%(local_rank, length, len(self.inputs)))
                logger.warning("Saving features into cached file %s", cached_file)
            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])

class EvalDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_blocksize_%d"%(block_size))
        if os.path.exists(cached_file) and not args.overwrite_cache:
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)

        else:
            self.inputs = []

            datafile = os.path.join(args.data_dir, f"{file_type}.txt")
            with open(datafile) as f:
                data = f.readlines()

            length = len(data)
            logger.info("Data size: %d"%(length))
            input_ids = []
            for idx,x in enumerate(data):
                x = x.strip()
                if x.startswith("<s>") and x.endswith("</s>"):
                    pass
                else:
                    x = "<s> " + x + " </s>"
                try:
                    input_ids.extend(tokenizer.encode(x))
                except Exception:
                    pass
                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.warning("load %d"%(percent))
            del data
            gc.collect()

            logger.info(f"tokens: {len(input_ids)}")
            self.split(input_ids, tokenizer, logger, block_size=block_size)
            del input_ids
            gc.collect()

            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def split(self, input_ids, tokenizer, logger, block_size=1024):
        sample = []
        i = 0
        while i < len(input_ids):
            sample = input_ids[i: i+block_size]
            if len(sample) == block_size:
                for j in range(block_size):
                    if tokenizer.convert_ids_to_tokens(sample[block_size-1-j])[0] == '\u0120' or tokenizer.convert_ids_to_tokens(sample[block_size-1-j]).startswith("<NUM_LIT"):
                        break
                    if sample[block_size-1-j] in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id]:
                        if sample[block_size-1-j] != tokenizer.bos_token_id:
                            j -= 1
                        break
                if j == block_size-1:
                    print(tokenizer.decode(sample))
                    exit()
                sample = sample[: block_size-1-j]
            # print(len(sample))
            i += len(sample)
            pad_len = block_size-len(sample)
            sample += [tokenizer.pad_token_id]*pad_len
            self.inputs.append(sample)

            if len(self.inputs) % 10000 == 0:
                logger.info(f"{len(self.inputs)} samples")


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])
        


class lineDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='test', block_size=924):
        datafile = os.path.join(args.data_dir, f"{file_type}.json")
        with open(datafile) as f:
            datas = f.readlines()

        length = len(datas)
        logger.info("Data size: %d"%(length))
        self.inputs = []
        self.gts = []
        for data in datas:
            data = json.loads(data.strip())
            self.inputs.append(tokenizer.encode(data["input"])[-block_size:])
            self.gts.append(data["gt"])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), self.gts[item]


class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    assert len(filename.split(','))==2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1,open(trg_filename) as f2:
            for line1,line2 in zip(f1,f2):
                examples.append(
                Example(
                        idx = idx,
                        source=line1.strip(),
                        target=line2.strip(),
                        )
                )
                idx+=1
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask

def convert_examples_to_features(examples, tokenizer, args, logger, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        #source
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        #    tokenizer.add_special_tokens({
        #        'pad_token': '<pad>',
        #        'cls_token': '<s>',
        #        'sep_token': '</s>'
        #    })
        #source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
        #source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_tokens = tokenizer.tokenize(example.source)[:args.block_size]

        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.block_size - len(source_ids)

        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length

        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            #target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
            target_tokens = tokenizer.tokenize(example.target)[:args.block_size]

        #target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]
        target_tokens = target_tokens

        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.block_size - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length

        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
    return features


class JavafinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, filename, file_type='train', max_length=512):
        self.max_length = max_length 
        self.tokenizer = tokenizer
        self.examples = self.read_examples(filename)

    def read_examples(self, filename):
        """Read examples from filename."""
        examples = {'source': [], 'target': []}
        assert len(filename.split(','))==2
        src_filename = filename.split(',')[0]
        trg_filename = filename.split(',')[1]
        with open(src_filename) as f1,open(trg_filename) as f2:
            for line1,line2 in zip(f1,f2):
                examples['source'].append(line1.strip()),
                examples['target'].append(line2.strip()),
        return examples

    def pack_samples(self, idx):
        """
        Repeatedly pick question, answer pairs from self.dataroot until we hit max_tokens.
        This will not include the tokens for the QUESTION and ANSWER prompt, as well as the  
        self.question_prefix. These will be added later and the total input will be 
        truncated if necessary.
        Always include the sample at idx at the beginning.
        """
        curr_num_tokens = 0
        curr_samples = [] 

        curr_q, curr_a = self.examples['source'][idx], self.examples['target'][idx]
        while curr_num_tokens < self.max_length:

            # Never remove. Fixes stalling bug.
            curr_q = curr_q[:150000]
            curr_a = curr_a[:150000]

            curr_num_tokens += len(self.tokenizer.tokenize(curr_q))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_a))

            curr_samples.append((curr_q, curr_a))

            random_idx = random.choice(range(len(self.examples['target'])))
            curr_q = self.examples['source'][random_idx]
            curr_a = self.examples['target'][random_idx] 

        return curr_samples

    def __len__(self):
        return min(len(self.examples['source']),
                   len(self.examples['target']))

    def __getitem__(self, idx):
        input_ids = []
        label_ids = []

        raw_samples = self.pack_samples(idx)
        for q_str, a_str in raw_samples: 
            q_str = self.tokenizer.cls_token + self.examples['source'][idx] + \
                self.tokenizer.sep_token
            a_str = self.examples['target'][idx]
            question_token_ids = self.tokenizer.encode(q_str, verbose=False)
            answer_token_ids   = self.tokenizer.encode(a_str, verbose=False)
            answer_token_ids.append(self.tokenizer.eos_token_id)
            input_ids.extend(question_token_ids)
            input_ids.extend(answer_token_ids)
            label_ids.extend([-100] * len(question_token_ids))
            label_ids.extend(answer_token_ids)

        # Cut off the excess
        input_ids = input_ids[:self.max_length]
        label_ids = label_ids[:self.max_length]

        retval = {
            "input_ids" : torch.LongTensor(input_ids),
            "labels" :  torch.LongTensor(label_ids)
        }
        gc.collect()
        return retval


if __name__ == '__main__':
    import argparse
    import transformers

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    data_prefix='/scratch1/08401/ywen/data/c2c_data'
    args.trainfile='{}/context_data.final,{}/body_data.final'.format(data_prefix, data_prefix)
    args.devfile='{}/context_data.final.100,{}/body_data.final.100'.format(data_prefix, data_prefix)
    args.output_dir = 'save/c2c_data_gptneo'
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        'EleutherAI/gpt-neo-125M',
        do_lower_case=False,
        sep_token='</s>', cls_token='<s>',
        pad_token='<pad>', unk_token='<|UNKNOWN|>')
    dataset = JavafinetuneDataset(tokenizer, args.devfile, max_length=512)
    #dev_dataset = JavafinetuneDataset(tokenizer, args, logger, file_type='dev', block_size=1024)

    e = dataset[0]
    print(e)
    print("------- input_ids ------------------------------------------------------------------------------------")
    print(tokenizer.decode(e['input_ids']))
    print("------- labels ------------------------------------------------------------------------------------")
    labels = e['labels']
    labels[labels == -100] = tokenizer.eos_token_id
    labels_str = tokenizer.decode(labels)
    print(labels_str)

    import pdb; pdb.set_trace()