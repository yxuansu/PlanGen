# coding=utf-8
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random
import numpy as np
import time
import logging
import progressbar

import logging
logging.getLogger('transformers.generation_utils').disabled = True

def parse_config():
    parser = argparse.ArgumentParser()
    # model configuration
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--crf_low_rank", type=int)
    parser.add_argument("--crf_beam_size", type=int)
    # data configuration
    parser.add_argument("--train_src_path", type=str)
    parser.add_argument("--train_tgt_path", type=str)
    parser.add_argument("--dev_src_path", type=str)
    parser.add_argument("--dev_tgt_path", type=str)
    parser.add_argument("--max_src_len", type=int)
    parser.add_argument("--max_tgt_len", type=int)
    parser.add_argument("--special_token_path", type=str)
    parser.add_argument("--min_slot_key_cnt", type=int)
    # mini-batch training configuration
    parser.add_argument("--number_of_gpu", type=int, help="Number of available GPUs.")  
    parser.add_argument("--batch_size_per_gpu", type=int, help='batch size for each gpu.') 
    parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation step.")
    parser.add_argument("--effective_batch_size", type=int, 
        help="effective_bsz = batch_size_per_gpu x number_of_gpu x gradient_accumulation_steps")
    # pre-training configuration
    parser.add_argument("--total_steps", type=int, 
        help="total effective training steps")
    parser.add_argument("--print_every", type=int, 
        help="how many update steps to print one intermediate result")
    parser.add_argument("--save_every", type=int, 
        help="how many update steps to save one model")
    # learning configuration
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--mle_loss_weight", type=float) # 0.5
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--save_path_prefix", type=str, help="directory to save the model parameters.")
    return parser.parse_args()

import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print ('Using single GPU training.')
    else:
        pass
    args = parse_config()
    device = torch.device('cuda')
    model_name = args.model_name
    print ('Initializaing model...')
    from utlis import load_special_tokens
    special_token_list = load_special_tokens(args.special_token_path, args.min_slot_key_cnt)
    from contentplanner import ContentPlanner
    model = ContentPlanner(model_name, args.crf_low_rank, args.crf_beam_size, special_token_list=special_token_list)
    print ('Model initialized!')

    print ('Loading data...')
    from dataclass import Data
    src_tokenizer = model.tokenizer
    tgt_tokenizer = model.targettokenizer
    data = Data(args.train_src_path, args.train_tgt_path, args.dev_src_path, args.dev_tgt_path, 
        src_tokenizer, tgt_tokenizer, args.max_src_len, args.max_tgt_len)
    print ('Data loaded.')

    from trainer import model_training
    print ('############################################################')
    print ('Start Training...')
    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    print ('Model loaded') 
    total_steps, print_every, save_every = args.total_steps, args.print_every, args.save_every
    ckpt_save_path = args.save_path_prefix
    model = model_training(args, data, model, total_steps, print_every, save_every, 
        ckpt_save_path, cuda_available, device)
    print ('Training stage completed!')
    print ('############################################################')
