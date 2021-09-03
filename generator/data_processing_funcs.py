import json
import progressbar
import numpy as np
from nltk import word_tokenize

import nltk
import re
from difflib import SequenceMatcher
from nltk.corpus import stopwords
stopword_set = set(stopwords.words('english'))

import multiprocessing as mp
####### ----- #######
UNSEEN_SLOT_KEY, EOS, SEP = '__None__', '__EOS__', '__SEP__'

def parse_subtable_metastr(one_json_dict, train_special_token_dict):
    one_sub_str = one_json_dict['subtable_metadata_str']
    
    ordered_cell_list = []
    if len(one_sub_str.split('<page_title>')) > 1:
        page_title_text = ' '.join(word_tokenize(one_sub_str.split('<page_title>')[1].split('</page_title>')[0].strip()))
    else:
        page_title_text = ''
    ordered_cell_list.append({"slot_key":'__page_title__', "slot_value":page_title_text})
            
    visited_cell_text_set = set()
    cell_list = one_sub_str.split('<cell>')[1:]
    for item in cell_list:
        cell_text = ' '.join(word_tokenize(item.split('<col_header>')[0].strip().split('</cell>')[0].strip().split('<row_header>')[0].strip()))
        if len(item.split('<col_header>')) > 1:
            all_possible_col_header_list = []
            candi_list = item.split('<col_header>')[1:]
            for one_candi in candi_list:
                one_candi_header = one_candi.split('</col_header>')[0].strip()
                all_possible_col_header_list.append(one_candi_header)

            nearest_col_header = '__None__'
            for one_candi in all_possible_col_header_list:
                if len(one_candi) > 0:
                    candi_nearest_col_header = '__' + '_'.join(one_candi.split()).strip('_') + '__'
                    if candi_nearest_col_header in train_special_token_dict:
                        nearest_col_header = candi_nearest_col_header
                        break
        else:
            nearest_col_header = '__None__'

        if nearest_col_header in train_special_token_dict:
            pass
        else: 
            nearest_col_header = '__None__' # for the non-overlap dataset
            
        if cell_text in visited_cell_text_set:
            pass
        else:
            ordered_cell_list.append({"slot_key":nearest_col_header, "slot_value":cell_text})
            visited_cell_text_set.add(cell_text)

    if len(one_sub_str.split('<section_title>')) > 1:
        section_title_text = ' '.join(word_tokenize(one_sub_str.split('<section_title>')[1].split('</section_title>')[0].strip()))
    else:
        section_title_text = ''
    ordered_cell_list.append({"slot_key":'__section_title__', "slot_value":section_title_text})
    
    return ordered_cell_list

def find_first_reference_sentence(one_json_dict):
    annotation_list = one_json_dict['sentence_annotations']
    references = []
    original_references = []
    length_list = []
    for item in annotation_list:
        one_final_sentence = ' '.join(word_tokenize(item['final_sentence']))
        one_len = len(one_final_sentence.split())
        length_list.append(one_len)
        references.append(one_final_sentence)
        original_references.append(item['final_sentence'])
    return references[0], original_references[0]

def process_one_json_dict(one_json_dict, special_token_dict):
    ordered_cell_list = parse_subtable_metastr(one_json_dict, special_token_dict)
    tokenized_reference, original_reference = find_first_reference_sentence(one_json_dict)
    return tokenized_reference, original_reference, ordered_cell_list

def map_content_to_order_dict(content_text):
    '''
        '__Governor__2__ __#__1__ __page_title__0__ __Took_Office__3__'
        {2:1, 1:2, 0:3, 3:4}
        source position to target position dictionary
    '''
    token_list = content_text.split()
    ref_len = len(token_list)
    map_dict = {}
    for tgt_pos in range(ref_len):
        one_token = token_list[tgt_pos]
        src_pos = int(one_token.strip('__').split('__')[-1])
        map_dict[src_pos] = tgt_pos
    return map_dict

END_OF_SLOT = '__EOS__'
def separate_parse_snippet(text, tokenizer, is_front):
    # is_front: whether this is the front of entire sequence
    token_list = text.strip().split()
    token_id_list = []
    for idx in range(len(token_list)):
        if idx == 0:
            if is_front:
                one_token = token_list[idx]
            else:
                one_token = ' ' + token_list[idx]
        else:
            one_token = ' ' + token_list[idx]
        one_token_id = tokenizer.encode(one_token, max_length=512, truncation=True, add_special_tokens=False)
        token_id_list += one_token_id
    return token_id_list

def restore_one_content_token(token):
    split_list = token.strip().strip('__').split('__')
    try:
        assert len(split_list) == 2
    except AssertionError:
        return '__None__'
    return '__' + split_list[0] + '__'

def restore_original_content_text(text):
    res_list = []
    for token in text.split():
        one_token = restore_one_content_token(token)
        res_list.append(one_token)
    return ' '.join(res_list).strip()


def join_cell_list_and_order_map(ordered_cell_list, map_dict, tokenizer, special_token_dict):
    #eos_idx = tokenizer.convert_tokens_to_ids([END_OF_SLOT])[0]
    '''
        ordered_cell_list:
            [{'slot_key': '__page_title__',
              'slot_value': 'List of Governors of South Carolina'},
             {'slot_key': '__section_title__',
              'slot_value': 'Governors under the Constitution of 1868'},
             {'slot_key': '__#__', 'slot_value': '76'},
             {'slot_key': '__Governor__', 'slot_value': 'Daniel Henry Chamberlain'},
             {'slot_key': '__Took_Office__', 'slot_value': 'December 1 , 1874'}]
            
        map_dict: mapping between source and target slot key position
            {2: 0, 1: 1, 0: 2, 3: 3}
    '''
    input_text = ''
    accumulated_len = 0
    for item in ordered_cell_list:
        one_text = item['slot_key'] + ' : ' + item['slot_value'] + ' ' + END_OF_SLOT
        one_len = len(tokenizer.encode(one_text, max_length=512, truncation=True, add_special_tokens=False))
        accumulated_len += one_len 
        if accumulated_len > 400:
            break
        input_text += one_text + ' '
    input_text = ' '.join(input_text.split()).strip()
    split_list = input_text.strip().split(END_OF_SLOT)[:-1]
    all_src_id_list, all_tgt_pos_list = [], []
    for idx in range(len(split_list)):
        one_snippet = split_list[idx]
        if idx == 0:
            is_front = True
        else:
            is_front = False
        one_snippet += END_OF_SLOT
        one_snippet_id_list = separate_parse_snippet(one_snippet, tokenizer, is_front)
        one_tgt_pos_list = []
        try:
            tgt_pos = map_dict[idx] + 1
        except KeyError:
            tgt_pos = 0
        for one_id in one_snippet_id_list:
            one_id_token = tokenizer.convert_ids_to_tokens([one_id])[0]
            #if one_id in special_id_set:
            if one_id_token in special_token_dict and one_id_token != END_OF_SLOT:
                one_tgt_pos_list.append(tgt_pos)
            else:
                one_tgt_pos_list.append(-1)
        all_src_id_list += one_snippet_id_list
        all_tgt_pos_list += one_tgt_pos_list
    #return all_src_id_list
    assert all_src_id_list == tokenizer.encode(input_text, max_length=512, truncation=True, add_special_tokens=False)
    assert len(all_src_id_list) == len(all_tgt_pos_list)
    return all_src_id_list, all_tgt_pos_list

