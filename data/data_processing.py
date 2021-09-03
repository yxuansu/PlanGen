import sys
import json
import progressbar
from data_processing_funcs import *
from text_processing_funcs import *

def load_special_tokens(special_token_path, min_cnt):
    special_token_list, special_token_dict = [], {}
    with open(special_token_path, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        for l in lines:
            content_list = l.strip('\n').split()
            token = content_list[0]
            cnt = int(content_list[1])
            if cnt >= min_cnt:
                special_token_list.append(token)
                special_token_dict[token] = 1
    print (len(special_token_list))
    return special_token_list, special_token_dict

def get_input_text(ordered_cell_list):
    input_text = ''
    for item in ordered_cell_list:
        one_text = item['slot_key'] + ' : ' + item['slot_value'] + ' ' + END_OF_SLOT
        input_text += one_text + ' '
    input_text = ' '.join(input_text.split()).strip()
    return input_text

def write_file(text_list, out_f):
    with open(out_f, 'w', encoding = 'utf8') as o:
        for text in text_list:
            o.writelines(text + '\n')

import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--special_token_path', type=str)
    parser.add_argument('--special_token_min_cnt', type=int)
    parser.add_argument('--raw_data_path', type=str)
    parser.add_argument('--file_head_name', type=str)
    parser.add_argument('--dataset_mode', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()
    special_token_path = args.special_token_path
    special_token_min_cnt = args.special_token_min_cnt
    special_token_list, special_token_dict = load_special_tokens(special_token_path, special_token_min_cnt)

    print ('Start loading raw data...')
    json_dict_list = []
    with open(args.raw_data_path, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        p = progressbar.ProgressBar(len(lines))
        p.start()
        idx = 0
        for l in lines:
            p.update(idx)
            idx += 1
            one_json_dict = json.loads(l.strip('\n'))
            json_dict_list.append(one_json_dict)
        p.finish()
    print ('Raw data loaded.')

    dataset_mode = args.dataset_mode
    print ('Processing data...')
    if dataset_mode == 'train' or dataset_mode == 'dev':
        all_src_text_list, all_reference_list = [], []
        all_content_text_list = []
        p = progressbar.ProgressBar(len(json_dict_list))
        p.start()
        idx = 0
        #for one_json_dict in [json_dict_list[710]]:
        for one_json_dict in json_dict_list:
            p.update(idx + 1)
            idx += 1
            #if idx == 10:
            #    break
            try:
                one_tokenized_reference, one_original_reference, one_ordered_cell_list = \
                process_one_json_dict(one_json_dict, special_token_dict)
                one_content_text = process_one_instance(one_tokenized_reference, one_ordered_cell_list)
                one_map_dict = map_content_to_order_dict(one_content_text)
                one_input_text = get_input_text(one_ordered_cell_list)
                
                all_src_text_list.append(one_input_text)
                all_reference_list.append(one_original_reference)
                one_content_text = restore_original_content_text(one_content_text)
                all_content_text_list.append(one_content_text)
            except:
                print (idx-1)
                pass
        p.finish()

        head_name = args.file_head_name
        table_file_name = head_name + '_table.txt'
        write_file(all_src_text_list, table_file_name)

        content_plan_file = head_name + '_content_plan.txt'
        write_file(all_content_text_list, content_plan_file)

        reference_file = head_name + '_reference.txt'
        write_file(all_reference_list, reference_file)

    elif dataset_mode == 'test':
        all_src_text_list = []
        p = progressbar.ProgressBar(len(json_dict_list))
        p.start()
        idx = 0
        #for one_json_dict in [json_dict_list[710]]:
        for one_json_dict in json_dict_list:
            p.update(idx + 1)
            idx += 1
            #if idx == 10:
            #    break
            try:
                one_ordered_cell_list = parse_subtable_metastr(one_json_dict, special_token_dict)
                one_input_text = get_input_text(one_ordered_cell_list)
                all_src_text_list.append(one_input_text)
            except:
                print (idx-1)
                pass
        p.finish()
        head_name = args.file_head_name
        table_file_name = head_name + '_table.txt'
        write_file(all_src_text_list, table_file_name)
    else:
        raise Exception('Wrong Dataset Mode!!!')
    print ('Data processed.')


