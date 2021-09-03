import sys
import torch
import random
import numpy as np
import json
from torch.nn.utils import rnn
import progressbar
from transformers import BartTokenizer, BartTokenizerFast, BartConfig
from transformers.modeling_bart import shift_tokens_right
from data_utlis import load_ordered_cell_list

UNSEEN_SLOT_KEY, EOS, SEP = '__None__', '__EOS__', '__SEP__'

class Data:
    def __init__(self, train_data_dict, dev_data_dict, max_table_len, max_content_plan_len, max_tgt_len, 
        model_name, special_token_path, min_slot_key_cnt, use_RL):

        self.max_table_len, self.max_content_plan_len, self.max_tgt_len = \
        max_table_len, max_content_plan_len, max_tgt_len
        self.special_token_list, self.special_token_dict = [], {}
        with open(special_token_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                one_special_token = l.strip('\n').split()[0]
                cnt = int(l.strip('\n').split()[1])
                if cnt >= min_slot_key_cnt:
                    self.special_token_list.append(one_special_token)
                    self.special_token_dict[one_special_token] = 1
                else:
                    pass
        print ('Number of Special Token is %d' % len(self.special_token_list))

        self.model_name = model_name
        self.tokenizer = BartTokenizerFast.from_pretrained(model_name)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.decode_tokenizer = BartTokenizer.from_pretrained(model_name)

        print ('original vocabulary Size %d' % len(self.tokenizer))
        self.tokenizer.add_tokens(self.special_token_list)
        self.decode_tokenizer.add_tokens(self.special_token_list)
        print ('vocabulary size after extension is %d' % len(self.tokenizer))

        self.sep_idx = self.tokenizer.convert_tokens_to_ids([SEP])[0]
        self.eos_idx = self.tokenizer.convert_tokens_to_ids([EOS])[0]
        print (self.eos_idx)

        print ('Start loading training data...')
        self.train_tabel_id_list, self.train_content_id_list, self.train_tgt_id_list, \
        self.train_id_content_dict, self.train_reference_text_list, \
        self.train_reference_content_plan_list = self.load_data(train_data_dict)
        if use_RL:
            print ('Loading Training Ordered Cell list...')
            train_processed_file_path = train_data_dict['processed_file_path']
            self.train_ordered_cell_list = load_ordered_cell_list(train_processed_file_path, self.special_token_dict)
        else:
            self.train_ordered_cell_list = [[] for _ in range(len(self.train_tabel_id_list))]
        print ('Training data loaded.')

        print ('Start loading validation data...')
        self.dev_tabel_id_list, self.dev_content_id_list, self.dev_tgt_id_list, \
        self.dev_id_content_dict, self.dev_reference_text_list, \
        self.dev_reference_content_plan_list = self.load_data(dev_data_dict)
        if use_RL:
            print ('Loading Validation Ordered Cell list...')
            dev_processed_file_path = dev_data_dict['processed_file_path']
            self.dev_ordered_cell_list = load_ordered_cell_list(dev_processed_file_path, self.special_token_dict)
        else:
            self.dev_ordered_cell_list = [[] for _ in range(len(self.dev_tabel_id_list))]
        print ('Validation data loaded.')

        self.train_num, self.dev_num = len(self.train_tabel_id_list), len(self.dev_tabel_id_list)
        print ('train number is %d, dev number is %d' % (self.train_num, self.dev_num))
        self.train_idx_list = [i for i in range(self.train_num)]
        self.dev_idx_list = [j for j in range(self.dev_num)]
        self.dev_current_idx = 0

    def load_one_text_id(self, text, max_len):
        text_id_list = self.tokenizer.encode(text, max_length=512, truncation=True, add_special_tokens=False)[:max_len]
        return text_id_list

    def load_text_id_list(self, path, max_len):
        text_list = []
        with open(path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                text_list.append(l.strip('\n'))

        p = progressbar.ProgressBar(len(text_list))
        p.start()
        res_id_list = []
        idx = 0
        for text in text_list:
            p.update(idx + 1)
            one_id_list = self.load_one_text_id(text, max_len)
            res_id_list.append(one_id_list)
            idx += 1
        p.finish()
        return res_id_list

    def load_data(self, data_dict):
        table_text_path = data_dict['table_text_path']
        print ('Loading Table Data...')
        tabel_id_list = self.load_text_id_list(table_text_path, self.max_table_len)
        tabel_id_list = [[self.sep_idx] + one_id_list for one_id_list in tabel_id_list]

        print ('Loading Content Data...')
        content_text_path = data_dict['content_text_path']
        content_id_list = self.load_text_id_list(content_text_path, self.max_content_plan_len)
        content_id_list = [[self.sep_idx] + one_id_list for one_id_list in content_id_list]
        assert len(tabel_id_list) == len(content_id_list)

        print ('Loading Reference Data...')
        reference_sentence_path = data_dict['reference_sentence_path']
        tgt_id_list = self.load_text_id_list(reference_sentence_path, self.max_tgt_len)
        assert len(tabel_id_list) == len(tgt_id_list)

        tgt_id_list = [[self.bos_token_id] + item + [self.eos_token_id] for item in tgt_id_list]

        id_content_dict = {}
        with open(content_text_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            idx = 0
            for l in lines:
                one_content_text = l.strip('\n')
                id_content_dict[idx] = one_content_text
                idx += 1

        reference_text_list = []
        with open(reference_sentence_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                one_text = l.strip('\n')
                reference_text_list.append(one_text)

        reference_content_plan_list = []
        with open(content_text_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                reference_content_plan_list.append(l.strip('\n'))
        return tabel_id_list, content_id_list, tgt_id_list, id_content_dict, reference_text_list, reference_content_plan_list

    def process_source_tensor(self, batch_src_id_list):
        batch_src_tensor_list = [torch.LongTensor(item) for item in batch_src_id_list]
        batch_src_tensor = rnn.pad_sequence(batch_src_tensor_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # ---- compute src mask ---- #
        batch_src_mask = torch.ones_like(batch_src_tensor)
        batch_src_mask = batch_src_mask.masked_fill(batch_src_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        return batch_src_tensor, batch_src_mask

    def process_decoder_tensor(self, batch_tgt_id_list):
        batch_tgt_tensor = [torch.LongTensor(item) for item in batch_tgt_id_list]
        batch_tgt_tensor = rnn.pad_sequence(batch_tgt_tensor, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_labels = batch_tgt_tensor
        batch_input = shift_tokens_right(batch_labels, self.tokenizer.pad_token_id)
        batch_labels[batch_labels[:, :] == self.tokenizer.pad_token_id] = -100
        return batch_input, batch_labels 

    def get_next_train_batch(self, batch_size):
        batch_idx_list = random.sample(self.train_idx_list, batch_size)
        batch_content_set = set()
        for one_idx in batch_idx_list:
            batch_content_set.add(self.train_id_content_dict[one_idx])

        # we ensure the same batch does not contain repeated content plan
        while len(batch_content_set) < batch_size:
            batch_idx_list = random.sample(self.train_idx_list, batch_size)
            batch_content_set = set()
            for one_idx in batch_idx_list:
                batch_content_set.add(self.train_id_content_dict[one_idx])

        batch_table_id_list, batch_content_id_list, batch_src_id_list, batch_tgt_id_list = [], [], [], []
        # data used for RL training
        batch_ordered_cell_list, batch_reference_text_list, batch_content_plan_list = [], [], []
        for idx in batch_idx_list:
            one_table_id_list = self.train_tabel_id_list[idx]
            one_content_id_list = self.train_content_id_list[idx]
            one_tgt_id_list = self.train_tgt_id_list[idx]

            batch_table_id_list.append(one_table_id_list)
            batch_content_id_list.append(one_content_id_list)
            batch_src_id_list.append(one_table_id_list + one_content_id_list)
            batch_tgt_id_list.append(one_tgt_id_list)

            one_ordered_cell_list = self.train_ordered_cell_list[idx]
            batch_ordered_cell_list.append(one_ordered_cell_list)
            one_reference_text = self.train_reference_text_list[idx]
            batch_reference_text_list.append(one_reference_text)
            one_content_plan = self.train_reference_content_plan_list[idx]
            batch_content_plan_list.append(one_content_plan)

        batch_table_tensor, batch_table_mask = self.process_source_tensor(batch_table_id_list)
        batch_content_tensor, batch_content_mask = self.process_source_tensor(batch_content_id_list)
        batch_src_tensor, batch_src_mask = self.process_source_tensor(batch_src_id_list)
        batch_tgt_in_tensor, batch_tgt_out_tensor = self.process_decoder_tensor(batch_tgt_id_list)
        return (batch_table_tensor, batch_table_mask), (batch_content_tensor, batch_content_mask), \
        (batch_src_tensor, batch_src_mask), (batch_tgt_in_tensor, batch_tgt_out_tensor), \
        (batch_ordered_cell_list, batch_reference_text_list, batch_content_plan_list)

    def get_next_dev_batch(self, batch_size):
        batch_table_id_list, batch_content_id_list, batch_src_id_list, batch_tgt_id_list = [], [], [], []
        batch_ordered_cell_list, batch_reference_text_list, batch_content_plan_list = [], [], []
        if self.dev_current_idx + batch_size < self.dev_num - 1:
            for i in range(batch_size):
                curr_idx = self.dev_current_idx + i
                one_table_id_list = self.dev_tabel_id_list[curr_idx]
                one_content_id_list = self.dev_content_id_list[curr_idx]
                one_tgt_id_list = self.dev_tgt_id_list[curr_idx]

                batch_table_id_list.append(one_table_id_list)
                batch_content_id_list.append(one_content_id_list)
                batch_src_id_list.append(one_table_id_list + one_content_id_list)
                batch_tgt_id_list.append(one_tgt_id_list)

                one_ordered_cell_list = self.dev_ordered_cell_list[curr_idx]
                batch_ordered_cell_list.append(one_ordered_cell_list)
                one_reference_text = self.dev_reference_text_list[curr_idx]
                batch_reference_text_list.append(one_reference_text)
                one_content_plan = self.dev_reference_content_plan_list[curr_idx]
                batch_content_plan_list.append(one_content_plan)
            self.dev_current_idx += batch_size
        else:
            for i in range(batch_size):
                curr_idx = self.dev_current_idx + i
                if curr_idx > self.dev_num - 1:  # 对dev_current_idx重新赋值
                    curr_idx = 0
                    self.dev_current_idx = 0
                else:
                    pass
                one_table_id_list = self.dev_tabel_id_list[curr_idx]
                one_content_id_list = self.dev_content_id_list[curr_idx]
                one_tgt_id_list = self.dev_tgt_id_list[curr_idx]

                batch_table_id_list.append(one_table_id_list)
                batch_content_id_list.append(one_content_id_list)
                batch_src_id_list.append(one_table_id_list + one_content_id_list)
                batch_tgt_id_list.append(one_tgt_id_list)

                one_ordered_cell_list = self.dev_ordered_cell_list[curr_idx]
                batch_ordered_cell_list.append(one_ordered_cell_list)
                one_reference_text = self.dev_reference_text_list[curr_idx]
                batch_reference_text_list.append(one_reference_text)
                one_content_plan = self.dev_reference_content_plan_list[curr_idx]
                batch_content_plan_list.append(one_content_plan)
            self.dev_current_idx = 0
        batch_table_tensor, batch_table_mask = self.process_source_tensor(batch_table_id_list)
        batch_content_tensor, batch_content_mask = self.process_source_tensor(batch_content_id_list)
        batch_src_tensor, batch_src_mask = self.process_source_tensor(batch_src_id_list)
        batch_tgt_in_tensor, batch_tgt_out_tensor = self.process_decoder_tensor(batch_tgt_id_list)
        return (batch_table_tensor, batch_table_mask), (batch_content_tensor, batch_content_mask), \
        (batch_src_tensor, batch_src_mask), (batch_tgt_in_tensor, batch_tgt_out_tensor), \
        (batch_ordered_cell_list, batch_reference_text_list, batch_content_plan_list)

