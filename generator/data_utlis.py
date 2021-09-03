import progressbar
from data_processing_funcs import *
from text_processing_funcs import *
import json

def load_ordered_cell_list(processed_file_path, special_token_dict):
    json_dict_list = []
    with open(processed_file_path, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        for l in lines:
            one_json_dict = json.loads(l.strip('\n'))
            json_dict_list.append(one_json_dict)

    p = progressbar.ProgressBar(len(json_dict_list))
    p.start()
    idx = 0
    ordered_cell_list = []
    for item in json_dict_list:
        idx += 1
        p.update(idx)
        one_ordered_cell_list = parse_subtable_metastr(item, special_token_dict)
        ordered_cell_list.append(one_ordered_cell_list)
    p.finish()
    print (len(ordered_cell_list))
    return ordered_cell_list


import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction

chencherry = SmoothingFunction()
def compute_corpus_bleu(list_of_target_text, list_of_pred_text):
    assert len(list_of_target_text) == len(list_of_pred_text)
    list_of_target_token_list, list_of_pred_token_list = [], []
    for k in range(len(list_of_target_text)):
        one_target_token_list = list_of_target_text[k].strip().split()
        list_of_target_token_list.append(one_target_token_list)
        one_pred_token_list = list_of_pred_text[k].strip().split()
        list_of_pred_token_list.append(one_pred_token_list)

    list_of_references, list_of_hypotheses = [], []
    for k in range(len(list_of_target_token_list)):
        list_of_references.append([list_of_target_token_list[k]])
        list_of_hypotheses.append(list_of_pred_token_list[k])
    return nltk.translate.bleu_score.corpus_bleu(list_of_references, 
            list_of_hypotheses, smoothing_function=chencherry.method1) * 100

def compute_sentence_bleu(reference_text, pred_text):
    references = [reference_text.split()]
    hypothesis = pred_text.split()
    return nltk.translate.bleu_score.sentence_bleu(references, hypothesis, smoothing_function=chencherry.method1)

def reward_estimation(tokenized_reference_text_list, reference_content_plan_list, pred_text_list, ordered_cell_list):
    assert len(tokenized_reference_text_list) == len(reference_content_plan_list)
    assert len(reference_content_plan_list) == len(pred_text_list)
    assert len(ordered_cell_list) == len(pred_text_list)

    batch_size = len(tokenized_reference_text_list) 
    tokenized_pred_text_list = []
    for text in pred_text_list:
        tokenized_pred_text_list.append(' '.join(word_tokenize(text.strip('\n'))))
    #sentence_bleu = compute_corpus_bleu(tokenized_reference_text_list, tokenized_pred_text_list)
    sentence_bleu_list = []
    for k in range(batch_size):
        one_ref_sen = tokenized_reference_text_list[k]
        one_pred_sen = tokenized_pred_text_list[k]
        one_sen_bleu = compute_sentence_bleu(one_ref_sen, one_pred_sen)
        sentence_bleu_list.append(one_sen_bleu)

    pred_content_plan_list = []
    content_plan_bleu_list = []
    for k in range(batch_size):
        one_ordered_cell = ordered_cell_list[k]
        one_pred_text = pred_text_list[k]
        one_pred_content_plan = process_one_instance(one_pred_text, one_ordered_cell)
        one_pred_content_plan = restore_original_content_text(one_pred_content_plan)
        one_content_plan_bleu = compute_sentence_bleu(reference_content_plan_list[k], one_pred_content_plan)
        content_plan_bleu_list.append(one_content_plan_bleu)
    return sentence_bleu_list, content_plan_bleu_list

def dev_corpus_bleu_estimation(tokenized_reference_text_list, reference_content_plan_list, pred_text_list, ordered_cell_list):
    assert len(tokenized_reference_text_list) == len(reference_content_plan_list)
    assert len(reference_content_plan_list) == len(pred_text_list)
    assert len(ordered_cell_list) == len(pred_text_list)

    batch_size = len(tokenized_reference_text_list) 
    tokenized_pred_text_list = []
    for text in pred_text_list:
        tokenized_pred_text_list.append(' '.join(word_tokenize(text.strip('\n'))))
    sentence_bleu = compute_corpus_bleu(tokenized_reference_text_list, tokenized_pred_text_list)

    pred_content_plan_list = []
    for k in range(batch_size):
        one_ordered_cell = ordered_cell_list[k]
        one_pred_text = pred_text_list[k]
        one_pred_content_plan = process_one_instance(one_pred_text, one_ordered_cell)
        one_pred_content_plan = restore_original_content_text(one_pred_content_plan)
        pred_content_plan_list.append(one_pred_content_plan)
    content_plan_bleu = compute_corpus_bleu(reference_content_plan_list, pred_content_plan_list)
    return sentence_bleu, content_plan_bleu

