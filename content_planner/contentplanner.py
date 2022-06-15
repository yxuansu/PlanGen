import torch
from torch import nn
from dynamic_crf_layer import DynamicCRF
from torch.nn import CrossEntropyLoss

train_fct = CrossEntropyLoss()
class TopLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, crf_low_rank, crf_beam_size, padding_idx):
        super(TopLayer, self).__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        self.crf_layer = DynamicCRF(num_embedding = vocab_size, low_rank = crf_low_rank, 
                                    beam_size = crf_beam_size)

        self.one_more_layer_norm = nn.LayerNorm(embed_dim)
        self.tgt_word_prj = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, src_representation, tgt_input):
        '''
            src_representation: bsz x seqlen x embed_dim
            tgt_input: bsz x seqlen 
        '''
        bsz, seqlen = tgt_input.size()
        src_representation = src_representation.transpose(0, 1) # seqlen x bsz x embed_dim
        src = src_representation

        emissions = self.tgt_word_prj(src.contiguous().view(-1, self.embed_dim)).view(seqlen, bsz, self.vocab_size)
        # compute mle loss
        logits = emissions.transpose(0,1).contiguous()
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        labels = tgt_input.clone()
        labels[labels[:, :] == self.padding_idx] = -100
        mle_loss = train_fct(logits.view(-1, self.vocab_size), labels.view(-1)) # averaged mle loss

        # compute crf loss
        emissions = emissions.transpose(0, 1) # [bsz x src_len x vocab_size]
        emission_mask = ~tgt_input.eq(self.padding_idx) # [bsz x src_len]
        batch_crf_loss = -1 * self.crf_layer(emissions, tgt_input, emission_mask) # [bsz]
        assert batch_crf_loss.size() == torch.Size([bsz])
        # create tgt mask
        tgt_mask = torch.ones_like(tgt_input)
        tgt_mask = tgt_mask.masked_fill(tgt_input.eq(self.padding_idx), 0.0).type(torch.FloatTensor)
        if tgt_input.is_cuda:
            tgt_mask = tgt_mask.cuda(tgt_input.get_device())
        crf_loss = torch.sum(batch_crf_loss) / torch.sum(tgt_mask)
        return mle_loss, crf_loss

    def decoding(self, src_representation):
        bsz, seqlen, _ = src_representation.size()
        src_representation = src_representation.transpose(0, 1) # seqlen x bsz x embed_dim
        src = src_representation

        emissions = self.tgt_word_prj(src.contiguous().view(-1, self.embed_dim)).view(seqlen, bsz, self.vocab_size)

        emissions = emissions.transpose(0, 1) # [bsz, seqlen, vocab_size]
        _, finalized_tokens = self.crf_layer.forward_decoder(emissions)
        assert finalized_tokens.size() == torch.Size([bsz, seqlen])
        return finalized_tokens

    def selective_decoding(self, src_representation, selective_mask):
        bsz, seqlen, _ = src_representation.size()
        src_representation = src_representation.transpose(0, 1) # seqlen x bsz x embed_dim
        src = src_representation

        emissions = self.tgt_word_prj(src.contiguous().view(-1, self.embed_dim)).view(seqlen, bsz, self.vocab_size)

        emissions = emissions.transpose(0, 1) # [bsz, seqlen, vocab_size]
        assert emissions.size() == selective_mask.size()
        emissions = emissions + selective_mask # mask the impossible token set

        _, finalized_tokens = self.crf_layer.forward_decoder(emissions)
        assert finalized_tokens.size() == torch.Size([bsz, seqlen])
        return finalized_tokens        

PAD_token = '[PAD]'
class ContentPlanner(nn.Module):
    def __init__(self, model_name, crf_low_rank=64, crf_beam_size=256, special_token_list=[]):
        super(ContentPlanner, self).__init__()
        from transformers import BertTokenizerFast, BertModel
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        if len(special_token_list) > 0:
            print ('Original vocabulary size is {}'.format(len(self.tokenizer)))
            print ('Adding special tokens...')
            self.tokenizer.add_tokens(special_token_list)
            print ('Special token added.')
            print ('Resizing language model embeddings...')
            self.model.resize_token_embeddings(len(self.tokenizer))
            print ('Language model embeddings resized.')
        self.vocab_size = len(self.tokenizer)
        print ('The vocabulary size of the language model is {}'.format(len(self.tokenizer)))
        self.embed_dim = self.model.config.hidden_size
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids([PAD_token])[0]

        from utlis import TargetTokenizer
        self.targettokenizer = TargetTokenizer(special_token_list, PAD_token)
        self.target_vocab_size = len(self.targettokenizer.vocab)

        self.toplayer = TopLayer(self.target_vocab_size, self.embed_dim, crf_low_rank, 
            crf_beam_size, self.targettokenizer.pad_token_id)

    def save_model(self, ckpt_save_path):
        import os
        if os.path.exists(ckpt_save_path):
            pass
        else: # recursively construct directory
            os.makedirs(ckpt_save_path, exist_ok=True)

        parameter_path = ckpt_save_path + '/parameters/'
        if os.path.exists(parameter_path):
            pass
        else: # recursively construct directory
            os.makedirs(parameter_path, exist_ok=True)

        torch.save({'model':self.state_dict(),
            'target_tokenizer': self.targettokenizer}, 
            parameter_path + r'model.bin')
        self.tokenizer.save_pretrained(ckpt_save_path)

    def load_pretrained_model(self, ckpt_save_path):
        print ('Loading pre-trained parameters...')
        parameter_path = ckpt_save_path + '/parameters/model.bin'
        if torch.cuda.is_available():
            print ('Cuda is available.')
            model_ckpt = torch.load(parameter_path)
        else:
            print ('Cuda is not available.')
            model_ckpt = torch.load(parameter_path, map_location='cpu')
        model_parameters = model_ckpt['model']
        self.load_state_dict(model_parameters)
        self.targettokenizer = model_ckpt['target_tokenizer']
        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained(ckpt_save_path)
        print ('Pre-trained parameters loaded!')

    def forward(self, src_input, tgt_input):
        '''
            src_input: bsz x seqlen
            tgt_input: bsz x seqlen
        '''
        bsz, seqlen = src_input.size()
        # create mask matrix
        src_mask = torch.ones_like(src_input)
        src_mask = src_mask.masked_fill(src_input.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        if src_input.is_cuda:
            src_mask = src_mask.cuda(src_input.get_device())

        outputs = self.model(input_ids=src_input, attention_mask=src_mask)
        src_representation = outputs[0]
        assert src_representation.size() == torch.Size([bsz, seqlen, self.embed_dim])
        mle_loss, crf_loss = self.toplayer(src_representation, tgt_input)
        return mle_loss, crf_loss

    def parse_one_output(self, id_list):
        short_list = []
        for idx in id_list:
            if idx == self.targettokenizer.eos_token_id:
                break
            else:
                short_list.append(idx)
        if len(short_list) == 0:
            short_list = [self.targettokenizer.unk_token_id]
        result = self.targettokenizer.convert_ids_to_text(short_list)
        return result 

    def parse_batch_output(self, finalized_tokens):
        predictions = finalized_tokens.detach().cpu().tolist()
        result = []
        for item in predictions:
            one_res = self.parse_one_output(item)
            result.append(one_res)
        return result

    def decode(self, src_input):
        bsz, seqlen = src_input.size()
        # create mask matrix
        src_mask = torch.ones_like(src_input)
        src_mask = src_mask.masked_fill(src_input.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        if src_input.is_cuda:
            src_mask = src_mask.cuda(src_input.get_device())

        outputs = self.model(input_ids=src_input, attention_mask=src_mask)
        src_representation = outputs[0]
        finalized_tokens = self.toplayer.decoding(src_representation)
        return self.parse_batch_output(finalized_tokens)

    # the part of selective decoding
    def produce_selective_mask(self, bsz, seqlen, vocab_size, selective_id_list):
        assert len(selective_id_list) == bsz
        res_list = []
        for idx in range(bsz):
            one_selective_id_list = selective_id_list[idx]
            one_tensor = torch.ones(vocab_size) * float('-inf')
            for s_id in one_selective_id_list:
                one_tensor[s_id] = 0.
            one_res = [one_tensor for _ in range(seqlen)]
            one_res = torch.stack(one_res, dim=0)
            assert one_res.size() == torch.Size([seqlen, vocab_size])
            res_list.append(one_res)
        res_mask = torch.stack(res_list, dim = 0)
        assert res_mask.size() == torch.Size([bsz, seqlen, vocab_size])
        return res_mask

    def selective_decoding(self, src_input, selective_id_list):
        '''
            selective_id_list: 
                A list of length bsz. Each item contains the selective ids of content plan. 
                The final generated path should be formatted with these selective ids. 
        '''
        bsz, seqlen = src_input.size()
        selective_mask = self.produce_selective_mask(bsz, seqlen, self.target_vocab_size, selective_id_list)
        if src_input.is_cuda:
            selective_mask = selective_mask.cuda(src_input.get_device())

        # create src mask matrix
        src_mask = torch.ones_like(src_input)
        src_mask = src_mask.masked_fill(src_input.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        if src_input.is_cuda:
            src_mask = src_mask.cuda(src_input.get_device())

        outputs = self.model(input_ids=src_input, attention_mask=src_mask)
        src_representation = outputs[0]
        finalized_tokens = self.toplayer.selective_decoding(src_representation, selective_mask)
        return self.parse_batch_output(finalized_tokens)
