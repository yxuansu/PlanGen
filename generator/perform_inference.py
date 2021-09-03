import nltk
nltk.download('stopwords')
nltk.download('punkt')
import os
import sys
import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import operator
from operator import itemgetter
from transformers import AdamW, get_linear_schedule_with_warmup
from utlis import eval_totto
import yaml
import subprocess
from subprocess import call
from data_utlis import reward_estimation, dev_corpus_bleu_estimation
import argparse
from dataclass import Data

def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--train_table_text_path', type=str)
    parser.add_argument('--train_content_text_path', type=str)
    parser.add_argument('--train_reference_sentence_path', type=str)
    parser.add_argument('--dev_table_text_path', type=str)
    parser.add_argument('--dev_content_text_path', type=str)
    parser.add_argument('--dev_reference_sentence_path', type=str)
    parser.add_argument('--dev_reference_path', type=str)
    parser.add_argument('--special_token_path', type=str)
    parser.add_argument('--max_table_len', type=int, default=320)
    parser.add_argument('--max_content_plan_len', type=int, default=25)
    parser.add_argument('--max_tgt_len', type=int, default=80)
    parser.add_argument('--min_slot_key_cnt', type=int, default=10)
    
    # model configuration
    parser.add_argument('--model_name', type=str, default='facebook/bart-base')
    parser.add_argument('--max_decode_len', type=int, default=80)
    # learning configuration
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pretrained_ckpt_path', type=str)
    return parser.parse_args()


def map_cuda(tensor_item, device, is_cuda):
    res_list = []
    if is_cuda:
        res_list.append(tensor_item[0].cuda(device))
        res_list.append(tensor_item[1].cuda(device))
    else:
        res_list = tensor_item
    return res_list

import argparse
if __name__ == '__main__':
    args = parse_config()
    device = args.gpu_id

    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()

    print ('Start loading data...')
    train_dict, dev_dict = {}, {}
    train_dict['table_text_path'] = args.train_table_text_path
    train_dict['reference_sentence_path'] = args.train_reference_sentence_path
    dev_dict['table_text_path'] = args.dev_table_text_path
    dev_dict['reference_sentence_path'] = args.dev_reference_sentence_path
    train_dict['content_text_path'] = args.train_content_text_path
    dev_dict['content_text_path'] = args.dev_content_text_path
    special_token_name = args.special_token_path

    train_dict['processed_file_path'] = None
    dev_dict['processed_file_path'] = None
    use_RL = False

    data = Data(train_dict, dev_dict, args.max_table_len, args.max_content_plan_len, args.max_tgt_len, 
        args.model_name, args.special_token_path, args.min_slot_key_cnt, use_RL)

    from generator import Generator
    model = Generator(model_name=args.model_name, tokenizer=data.decode_tokenizer, 
            max_decode_len=args.max_decode_len, dropout=0.0)

    print ('Loading Pretrained Parameters...')
    if torch.cuda.is_available():
        model_ckpt = torch.load(args.pretrained_ckpt_path)
    else:
        model_ckpt = torch.load(args.pretrained_ckpt_path, map_location='cpu')
    model_parameters = model_ckpt['model']
    model.load_state_dict(model_parameters)
    if torch.cuda.is_available():
        model = model.cuda(device)
    model.eval()
    print ('Model loaded.')

    dev_num = data.dev_num
    batch_size = args.batch_size
    dev_step_num = int(dev_num / batch_size) + 1

    os.makedirs(r'./inference_result/', exist_ok=True)
    dev_output_text_list = []
    # Dev RL estimation variables
    dev_sample_trajectory_list, dev_reference_text_list, dev_reference_content_plan_list, \
    dev_reference_ordered_cell_list = [], [], [], []
    with torch.no_grad():
        import progressbar
        p = progressbar.ProgressBar(dev_step_num)
        p.start()
        print ('Start evaluation...')
        for dev_step in range(dev_step_num):
            p.update(dev_step)
            _, _, dev_batch_src_item, dev_batch_tgt_item, _ = data.get_next_dev_batch(batch_size)

            cuda_available = torch.cuda.is_available()
            dev_batch_src_tensor, dev_batch_src_mask = map_cuda(dev_batch_src_item, device, cuda_available)
            dev_batch_tgt_in_tensor, dev_batch_tgt_out_tensor = map_cuda(dev_batch_tgt_item, device, cuda_available)
            # evaluation part
            # --- greedy decoding part --- #
            decoded_result = model.generate(dev_batch_src_tensor, dev_batch_src_mask)
            dev_output_text_list += decoded_result
        p.finish()

        dev_output_text_list = dev_output_text_list[:dev_num]
        dev_text_out_path = r'./inference_result/inference_test_out.txt'
        with open(dev_text_out_path, 'w', encoding = 'utf8') as o:
            for text in dev_output_text_list:
                o.writelines(text + '\n')

        overall_bleu, overlap_bleu, nonoverlap_bleu = eval_totto(dev_text_out_path, args.dev_reference_path)
        print ('Overall bleu is %5f, Overlap bleu is %5f, Nonoverlap bleu is %5f' % (overall_bleu, overlap_bleu, nonoverlap_bleu))
    