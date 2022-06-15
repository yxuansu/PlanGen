import nltk
import torch
from nltk.translate.bleu_score import SmoothingFunction

chencherry = SmoothingFunction()
def measure_bleu_score(prediction_list, reference_list):
    list_of_hypotheses = []
    for text in prediction_list:
        list_of_hypotheses.append(text.strip().split())

    list_of_references = []
    for text in reference_list:
        references = [text.strip().split()]
        list_of_references.append(references)

    score = nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses, 
                                      weights=(0.5, 0.5, 0.0, 0.0),
                                      smoothing_function=chencherry.method4)
    return round(score * 100, 2)


def load_special_tokens(special_token_path, min_slot_key_cnt):
    special_token_list = []
    with open(special_token_path, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        for l in lines:
            one_special_token = l.strip('\n').split()[0]
            cnt = int(l.strip('\n').split()[1])
            if cnt >= min_slot_key_cnt:
                special_token_list.append(one_special_token)
            else:
                pass
    print ('Number of Special Tokens is {}'.format(len(special_token_list)))
    return special_token_list

EOS_token = '__EOS__'
class TargetTokenizer:
    def __init__(self, token_list, pad_token):
        if len(token_list) == 0:
            token_list = ['__None__', '__EOS__']
        if pad_token in token_list:
            pass
        else:
            token_list = token_list + [pad_token]

        self.vocab = set()
        self.token_to_id_dict = {}
        self.id_to_token_dict = {}
        idx = 0
        for token in token_list:
            self.token_to_id_dict[token] = idx
            self.id_to_token_dict[idx] = token
            self.vocab.add(token)
            idx += 1

        self.unk_token = '__None__'
        self.unk_token_id = self.token_to_id_dict[self.unk_token]
        self.eos_token = '__EOS__'
        self.eos_token_id = self.token_to_id_dict[self.eos_token]
        self.pad_token = pad_token
        self.pad_token_id = self.token_to_id_dict[self.pad_token]
        print ('The target-side tokenizer vocabulary size is {}'.format(len(self.token_to_id_dict)))
        #print ('The unk token is {}, the unk token id is {}'.format(self.unk_token, self.unk_token_id))

    def convert_tokens_to_ids(self, token_list):
        id_list = []
        for token in token_list:
            if token in self.vocab:
                id_list.append(self.token_to_id_dict[token])
            else:
                id_list.append(self.unk_token_id)
        return id_list

    def convert_ids_to_tokens(self, id_list):
        token_list = []
        for idx in id_list:
            if idx < len(self.vocab):
                token_list.append(self.id_to_token_dict[int(idx)])
            else:
                token_list.append(self.unk_token)
        return token_list

    def convert_text_to_ids(self, text):
        token_list = text.strip().split()
        return self.convert_tokens_to_ids(token_list)

    def convert_ids_to_text(self, id_list):
        token_list = self.convert_ids_to_tokens(id_list)
        return ' '.join(token_list).strip()

    def extract_selective_ids(self, table_text):
        token_list = table_text.strip('\n').strip().split()
        selective_id_set = set()
        for token in token_list:
            if token.startswith(r'__') and token.endswith(r'__'):
                one_id = self.convert_tokens_to_ids([token])[0]
                selective_id_set.add(one_id)
            else:
                pass
        selective_id_set.add(self.eos_token_id)
        return list(selective_id_set)

