import json
import random
import torch
import numpy as np
import progressbar
from torch.nn.utils import rnn

PAD_token = '[PAD]'
class Data:
    def __init__(self, train_src_path, train_tgt_path, dev_src_path, dev_tgt_path, src_tokenizer, 
        tgt_tokenizer, max_src_len, max_tgt_len):
        # initialization 
        self.src_tokenizer, self.tgt_tokenizer = src_tokenizer, tgt_tokenizer
        self.src_pad_token_id = self.src_tokenizer.convert_tokens_to_ids([PAD_token])[0]
        self.tgt_pad_token_id = self.tgt_tokenizer.token_to_id_dict[PAD_token]
        self.tgt_eos_token_id = self.tgt_tokenizer.eos_token_id
        self.max_src_len, self.max_tgt_len = max_src_len, max_tgt_len
        self.cls_token_id, self.sep_token_id = self.src_tokenizer.cls_token_id, \
        self.src_tokenizer.sep_token_id
        # load data
        self.train_src_id_list, self.train_tgt_id_list, self.train_selective_id_list = \
        self.load_data(train_src_path, train_tgt_path)
        self.dev_src_id_list, self.dev_tgt_id_list, self.dev_selective_id_list = \
        self.load_data(dev_src_path, dev_tgt_path)

        self.train_num, self.dev_num = len(self.train_src_id_list), len(self.dev_src_id_list)
        print ('train number:{}, dev number:{}'.format(self.train_num, self.dev_num))

        self.train_idx_list = [i for i in range(self.train_num)]
        random.shuffle(self.train_idx_list)
        self.dev_idx_list = [j for j in range(self.dev_num)]
        self.dev_current_idx = 0

    def load_one_table_id(self, text):
        table_id_list = self.src_tokenizer.encode(text, max_length=self.max_src_len, truncation=True, 
            add_special_tokens=False)[:self.max_src_len]
        table_id_list = [self.cls_token_id] + table_id_list + [self.sep_token_id]
        return table_id_list

    def load_one_content_plan_id(self, text):
        token_list = text.strip().split()[:self.max_tgt_len]
        content_plan_id_list = self.tgt_tokenizer.convert_tokens_to_ids(token_list)
        content_plan_id_list = content_plan_id_list + \
        [self.tgt_eos_token_id, self.tgt_eos_token_id] # two eos tokens at the end
        return content_plan_id_list

    def load_data(self, src_path, tgt_path):
        with open(src_path, 'r', encoding = 'utf8') as i:
            src_lines = i.readlines()
        with open(tgt_path, 'r', encoding = 'utf8') as i:
            tgt_lines = i.readlines()
        assert len(src_lines) == len(tgt_lines)

        src_id_list, tgt_id_list, selective_id_list = [], [], []
        data_num = len(src_lines)
        p = progressbar.ProgressBar(data_num)
        p.start()
        for idx in range(data_num):
            p.update(idx)
            one_table_text = src_lines[idx].strip('\n').strip()
            one_table_id_list = self.load_one_table_id(one_table_text)
            one_content_plan_text = tgt_lines[idx].strip('\n').strip()
            one_content_plan_id_list = self.load_one_content_plan_id(one_content_plan_text)
            one_selective_id_list = self.tgt_tokenizer.extract_selective_ids(one_table_text.strip('\n').strip())
            src_id_list.append(one_table_id_list)
            tgt_id_list.append(one_content_plan_id_list)
            selective_id_list.append(one_selective_id_list)
        p.finish()
        return src_id_list, tgt_id_list, selective_id_list

    def pad_src_data(self, batch_id_list):
        batch_id_list = [torch.LongTensor(item) for item in batch_id_list]
        batch_tensor = rnn.pad_sequence(batch_id_list, batch_first=True, padding_value=self.src_pad_token_id)
        return batch_tensor

    def pad_tgt_data(self, batch_id_list, max_len):
        res_id_list = []
        for item in batch_id_list:
            pad_len = max_len - len(item)
            pad_id_list = [self.tgt_pad_token_id for _ in range(pad_len)]
            one_res_id_list = item + pad_id_list
            assert len(one_res_id_list) == max_len
            res_id_list.append(one_res_id_list)
        return torch.LongTensor(res_id_list)

    def get_next_train_batch(self, batch_size):
        batch_idx_list = random.sample(self.train_idx_list, batch_size)
        batch_src_id_list, batch_tgt_id_list, batch_selective_id_list = [], [], []

        for idx in batch_idx_list:
            batch_src_id_list.append(self.train_src_id_list[idx])
            batch_tgt_id_list.append(self.train_tgt_id_list[idx])
            batch_selective_id_list.append(self.train_selective_id_list[idx])

        batch_src_tensor = self.pad_src_data(batch_src_id_list)
        _, max_len = batch_src_tensor.size()
        batch_tgt_tensor = self.pad_tgt_data(batch_tgt_id_list, max_len)
        assert batch_src_tensor.size() == batch_tgt_tensor.size()
        return batch_src_tensor, batch_tgt_tensor, batch_selective_id_list

    def get_next_validation_batch(self, batch_size):
        curr_select_idx, instance_num = self.dev_current_idx, self.dev_num
        src_id_list, tgt_id_list, selective_id_list = \
        self.dev_src_id_list, self.dev_tgt_id_list, self.dev_selective_id_list

        batch_src_id_list, batch_tgt_id_list, batch_selective_id_list = [], [], []
        if curr_select_idx + batch_size < instance_num:
            for i in range(batch_size):
                curr_idx = curr_select_idx + i
                batch_src_id_list.append(src_id_list[curr_idx])
                batch_tgt_id_list.append(tgt_id_list[curr_idx])
                batch_selective_id_list.append(selective_id_list[curr_idx])
            self.dev_current_idx += batch_size
        else:
            for i in range(batch_size):
                curr_idx = curr_select_idx + i
                if curr_idx > instance_num - 1: 
                    curr_idx = 0
                    self.dev_current_idx = 0
                batch_src_id_list.append(src_id_list[curr_idx])
                batch_tgt_id_list.append(tgt_id_list[curr_idx])
                batch_selective_id_list.append(selective_id_list[curr_idx])
            self.dev_current_idx = 0
        batch_src_tensor = self.pad_src_data(batch_src_id_list)
        _, max_len = batch_src_tensor.size()
        batch_tgt_tensor = self.pad_tgt_data(batch_tgt_id_list, max_len)
        assert batch_src_tensor.size() == batch_tgt_tensor.size()
        return batch_src_tensor, batch_tgt_tensor, batch_selective_id_list
