import torch
from torch import nn
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

class Generator(nn.Module):
    def __init__(self, model_name, tokenizer, max_decode_len, dropout):
        super().__init__()
        self.tokenizer = tokenizer # tokenizer with extended vocabulary
        self.max_decode_len = max_decode_len

        print ('Initializing Huggingface BART model...')
        bart_config = BartConfig.from_pretrained(model_name)
        bart_config.__dict__["dropout"] = dropout
        #model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)
        self.model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)
        print ('Resizing Token Embeddings...')
        self.model.resize_token_embeddings(len(self.tokenizer)) 

        self.vocab_size = len(self.tokenizer)
        self.logsftmax = nn.LogSoftmax(dim=-1)
        self.padding_idx = self.tokenizer.pad_token_id

    def forward(self, src_input, src_mask, tgt_input, tgt_output):
        src_mask = src_mask.type(src_input.type())
        outputs = self.model(input_ids=src_input, attention_mask=src_mask, decoder_input_ids=tgt_input, labels=tgt_output)
        loss = outputs[0]#.mean()
        return loss

    def generate(self, src_input, src_mask):
        result_list = []
        outputs = self.model.generate(input_ids=src_input, attention_mask=src_mask, max_length=self.max_decode_len)
        for predicted_ids in outputs:
            one_result = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
            result_list.append(one_result)
        return result_list

    def RL_sampling(self, src_input, src_mask, top_k, temperature):
        sample_output = self.model.generate(input_ids=src_input, attention_mask=src_mask, do_sample=True, 
                                             max_length=self.max_decode_len, top_k=top_k, temperature=temperature)
        sample_input = sample_output[:, :-1].contiguous()
        sample_labels = sample_output[:, 1:].contiguous()
        bsz, sample_len = sample_input.size()
        # keep track of decoded result
        decoded_result_list = []
        for predicted_ids in sample_labels:
            one_result = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
            decoded_result_list.append(one_result)

        # get sampled loglikelihood
        outputs = self.model(input_ids=src_input, attention_mask=src_mask, decoder_input_ids=sample_input, 
                              labels=sample_labels, return_dict=True)
        logits = outputs[1]
        assert logits.size() == torch.Size([bsz, sample_len, self.vocab_size]) # un-softmax logits
        logprobs = self.logsftmax(logits)
        unsequeeze_sample_labels = sample_labels.unsqueeze(-1) # bsz x sample_len x 1
        gathered_logprobs = torch.gather(logprobs, dim = -1, index=unsequeeze_sample_labels).squeeze(-1)
        gathered_logprobs = gathered_logprobs.masked_fill(sample_labels.eq(self.padding_idx), float(0.0))
        track_gathered_logprobs = gathered_logprobs

        indicator_matrix = torch.ones_like(gathered_logprobs).type(gathered_logprobs.type())
        indicator_matrix = indicator_matrix.masked_fill(sample_labels.eq(self.padding_idx), float(0.0))

        gathered_logprobs = gathered_logprobs * indicator_matrix
        #return track_gathered_logprobs, indicator_matrix, gathered_logprobs, decoded_result_list, sample_output
        return gathered_logprobs, indicator_matrix, decoded_result_list

