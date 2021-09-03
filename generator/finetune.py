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
    parser.add_argument('--train_processed_file_path', type=str)
    parser.add_argument('--dev_table_text_path', type=str)
    parser.add_argument('--dev_content_text_path', type=str)
    parser.add_argument('--dev_reference_sentence_path', type=str)
    parser.add_argument('--dev_processed_file_path', type=str)
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
    parser.add_argument('--total_steps', type=int, default=200000)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--print_every', type=int, default=200)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--dropout', type=float, default=0.2)
    # RL configuration
    parser.add_argument('--pretrained_ckpt_path', type=str, default='./ckpt/pretrain/generator-pretrain.ckpt')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/finetune/')
    parser.add_argument('--RL_topk', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.5)
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

    test_output_dir = args.ckpt_path
    import os
    if os.path.exists(test_output_dir):
        pass
    else: # recursively construct directory
        os.makedirs(test_output_dir, exist_ok=True)
    os.makedirs(r'./rl_intermediate_result/', exist_ok=True)

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
    train_dict['processed_file_path'] = args.train_processed_file_path
    dev_dict['processed_file_path'] = args.dev_processed_file_path

    use_RL = True
    data = Data(train_dict, dev_dict, args.max_table_len, args.max_content_plan_len, args.max_tgt_len, 
        args.model_name, args.special_token_path, args.min_slot_key_cnt, use_RL)

    from generator import Generator
    model = Generator(model_name=args.model_name, tokenizer=data.decode_tokenizer, 
            max_decode_len=args.max_decode_len, dropout=args.dropout)

    print ('Loading Pretrained Parameters...')
    if torch.cuda.is_available():
        model_ckpt = torch.load(args.pretrained_ckpt_path)
    else:
        model_ckpt = torch.load(args.pretrained_ckpt_path, map_location='cpu')
    model_parameters = model_ckpt['model']
    model.load_state_dict(model_parameters)
    if torch.cuda.is_available():
        model = model.cuda(device)
    model.train()
    print ('Model loaded.')

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_update_steps = (args.total_steps // args.gradient_accumulation_steps) + 1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, 
            num_training_steps=total_update_steps)
    optimizer.zero_grad()

    train_num, dev_num = data.train_num, data.dev_num
    batch_size = args.batch_size
    train_step_num = int(train_num / batch_size) + 1
    dev_step_num = int(dev_num / batch_size) + 1

    batches_processed = 0
    max_dev_score = 0.
    total_steps = args.total_steps
    print_every, eval_every = args.print_every, args.eval_every



    model.train()

    # RL configuration
    RL_topk, temperature = args.RL_topk, args.temperature
    for one_step in range(total_steps):
        epoch = one_step // train_step_num
        batches_processed += 1

        train_MLE_loss_accumulated, train_RL_loss_accumulated = 0., 0.

        _, _, train_batch_src_item, train_batch_tgt_item, train_batch_RL_input = data.get_next_train_batch(batch_size)
        train_batch_ordered_cell_list, train_batch_reference_text_list, train_batch_content_plan_list = \
        train_batch_RL_input

        train_batch_src_tensor, train_batch_src_mask = map_cuda(train_batch_src_item, device, cuda_available)
        train_batch_tgt_in_tensor, train_batch_tgt_out_tensor = map_cuda(train_batch_tgt_item, device, cuda_available)
        
        # compute MLE loss
        train_generation_loss = model(train_batch_src_tensor, train_batch_src_mask, train_batch_tgt_in_tensor, train_batch_tgt_out_tensor)
        train_MLE_loss_accumulated += train_generation_loss.item()

        ##########################################################################################################################
        # RL loss
        '''
            train_gathered_logprobs: log probability matrix of sampled trajectories; bsz x sample_len
            train_indicator_matrix: indicating which entry in the gathered log prob matrix is valid;; bsz x sample_len
        '''
        train_gathered_logprobs, train_indicator_matrix, train_trajectory_list = \
        model.RL_sampling(train_batch_src_tensor, train_batch_src_mask, top_k=RL_topk, temperature=temperature)
        # measure reward
        train_sentence_bleu_list, train_content_plan_bleu_list = reward_estimation(train_batch_reference_text_list, 
        train_batch_content_plan_list, train_trajectory_list, train_batch_ordered_cell_list)
        assert len(train_sentence_bleu_list) == batch_size
        train_sentence_reward = torch.FloatTensor(train_sentence_bleu_list).type(train_indicator_matrix.type()).unsqueeze(-1)
        assert train_sentence_reward.size() == torch.Size([batch_size, 1])
        train_content_plan_reward = torch.FloatTensor(train_content_plan_bleu_list).type(train_indicator_matrix.type()).unsqueeze(-1)
        train_reward = train_sentence_reward + train_content_plan_reward
        train_sample_logprobs = train_gathered_logprobs * train_indicator_matrix
        train_RL_term = train_reward * train_sample_logprobs
        train_RL_loss = (-1 * torch.sum(train_RL_term)) / torch.sum(train_indicator_matrix)
        train_RL_loss_accumulated += train_RL_loss.item()
        ##########################################################################################################################
        
        train_loss = train_generation_loss + train_RL_loss
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if (one_step+1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if batches_processed % print_every == 0:
            curr_train_MLE_loss = train_MLE_loss_accumulated / print_every
            curr_train_RL_loss = train_RL_loss_accumulated / print_every
            print ('At epoch %d, batch %d, MLE loss %.5f, RL loss %.5f, max combine score is %5f' % 
                (epoch, batches_processed, curr_train_MLE_loss, curr_train_RL_loss, max_dev_score))
            train_MLE_loss_accumulated, train_RL_loss_accumulated = 0., 0.

            # keep track of intermediate sampled result
            train_sample_out_path = r'./rl_intermediate_result/train_sample_out.txt'
            with open(train_sample_out_path, 'a', encoding = 'utf8') as o:
                train_sample_number = len(train_trajectory_list)
                assert len(train_trajectory_list) == len(train_batch_reference_text_list)
                for m in range(train_sample_number):
                    o.writelines(train_batch_reference_text_list[m] + '\n')
                    o.writelines(train_trajectory_list[m] + '\n')
                    o.writelines('\n')
                o.writelines('------------------------------------------' + '\n')


        if batches_processed % eval_every == 0:
            model.eval()
            dev_MLE_loss_accumulated, dev_RL_loss_accumulated = 0., 0.
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
                    _, _, dev_batch_src_item, dev_batch_tgt_item, dev_batch_RL_input = data.get_next_dev_batch(batch_size)
                    dev_batch_ordered_cell_list, dev_batch_reference_text_list, dev_batch_content_plan_list = \
                    dev_batch_RL_input

                    cuda_available = torch.cuda.is_available()
                    dev_batch_src_tensor, dev_batch_src_mask = map_cuda(dev_batch_src_item, device, cuda_available)
                    dev_batch_tgt_in_tensor, dev_batch_tgt_out_tensor = map_cuda(dev_batch_tgt_item, device, cuda_available)

                    dev_generation_loss = model(dev_batch_src_tensor, dev_batch_src_mask, 
                        dev_batch_tgt_in_tensor, dev_batch_tgt_out_tensor)

                    dev_MLE_loss_accumulated += dev_generation_loss.item()
                    ##########################################################################################################################
                    # RL loss
                    dev_gathered_logprobs, dev_indicator_matrix, dev_trajectory_list = \
                    model.RL_sampling(dev_batch_src_tensor, dev_batch_src_mask, top_k=RL_topk, temperature=temperature)
                    # measure reward
                    dev_sentence_bleu_list, dev_content_plan_bleu_list = reward_estimation(dev_batch_reference_text_list, 
                        dev_batch_content_plan_list, dev_trajectory_list, dev_batch_ordered_cell_list)
                    assert len(dev_sentence_bleu_list) == batch_size
                    dev_sentence_reward = torch.FloatTensor(dev_sentence_bleu_list).type(dev_indicator_matrix.type()).unsqueeze(-1)
                    assert dev_sentence_reward.size() == torch.Size([batch_size, 1])
                    dev_content_plan_reward = torch.FloatTensor(dev_content_plan_bleu_list).type(dev_indicator_matrix.type()).unsqueeze(-1)
                    dev_reward = dev_sentence_reward + dev_content_plan_reward
                    dev_sample_logprobs = dev_gathered_logprobs * dev_indicator_matrix
                    dev_RL_term = dev_reward * dev_sample_logprobs
                    dev_RL_loss = (-1 * torch.sum(dev_RL_term)) / torch.sum(dev_indicator_matrix)
                    dev_RL_loss_accumulated += dev_RL_loss.item()
                    ##########################################################################################################################
                    
                    # evaluation part
                    # --- greedy decoding part --- #
                    decoded_result = model.generate(dev_batch_src_tensor, dev_batch_src_mask)
                    dev_output_text_list += decoded_result
                    # --- RL sampling part --- #
                    dev_sample_trajectory_list += dev_trajectory_list
                    dev_reference_text_list += dev_batch_reference_text_list
                    dev_reference_content_plan_list += dev_batch_content_plan_list
                    dev_reference_ordered_cell_list += dev_batch_ordered_cell_list
                p.finish()

                dev_output_text_list = dev_output_text_list[:dev_num]
                dev_text_out_path = r'./rl_intermediate_result/test_out.txt'
                with open(dev_text_out_path, 'w', encoding = 'utf8') as o:
                    for text in dev_output_text_list:
                        o.writelines(text + '\n')
                # --- RL search evaluation --- #
                dev_sample_trajectory_list = dev_sample_trajectory_list[:dev_num]
                dev_reference_text_list = dev_reference_text_list[:dev_num]
                dev_reference_content_plan_list = dev_reference_content_plan_list[:dev_num]
                dev_reference_ordered_cell_list = dev_reference_ordered_cell_list[:dev_num]
                dev_sample_out_path = r'./rl_intermediate_result//sample_out.txt'
                with open(dev_sample_out_path, 'w', encoding = 'utf8') as o:
                    for text in dev_sample_trajectory_list:
                        o.writelines(text + '\n')

                overall_bleu, overlap_bleu, nonoverlap_bleu = eval_totto(dev_text_out_path, args.dev_reference_path)
                # --- RL search evaluation --- #
                _, dev_corpus_content_bleu = dev_corpus_bleu_estimation(dev_reference_text_list, 
                        dev_reference_content_plan_list, dev_output_text_list, dev_reference_ordered_cell_list)

                one_dev_combine_score = overall_bleu + overlap_bleu + nonoverlap_bleu
                one_dev_MLE_loss = dev_MLE_loss_accumulated / dev_step_num
                one_dev_RL_loss = dev_RL_loss_accumulated / dev_step_num
                print ('----------------------------------------------------------------')
                print ('At epoch %d, batch %d, overall content plan bleu is %5f, overall bleu is %5f, overlap bleu is %5f, \
                    nonoverlap bleu is %5f, combine_score is %.3f, dev MLE loss is %5f, dev RL loss is %5f' \
                    % (epoch, batches_processed, dev_corpus_content_bleu, overall_bleu, overlap_bleu, nonoverlap_bleu, \
                        round(one_dev_combine_score, 3), one_dev_MLE_loss, one_dev_RL_loss))
                save_name = r'generator-rl-finetune.ckpt'
                print ('----------------------------------------------------------------')

                if one_dev_combine_score > max_dev_score:
                    torch.save({'model':model.state_dict()}, test_output_dir + save_name)
                    max_dev_score = one_dev_combine_score
                else:
                    pass
                fileData = {}
                for fname in os.listdir(test_output_dir):
                    if fname.startswith('generator'):
                        fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                    else:
                        pass
                sortedFiles = sorted(fileData.items(), key=itemgetter(1))
                if len(sortedFiles) < 1:
                    pass
                else:
                    delete = len(sortedFiles) - 1
                    for x in range(0, delete):
                        os.remove(test_output_dir + '/' + sortedFiles[x][0])
            model.train()

    