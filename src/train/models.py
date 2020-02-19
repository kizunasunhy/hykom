# -*- coding: utf-8 -*-


"""
author: Sun Hongyang,
email: kizuna.sunhy@gmail.com
"""


###########
# imports #
###########
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
#import fasttext

from src.rsc.resource import Resource
from torch.nn import utils as nn_utils
from torch.autograd import Variable
from torch import FloatTensor, Tensor


#########
# types #
#########
class PosModel(nn.Module):
    """
    part-of-speech tagger pytorch model
    """
    def __init__(self, cfg: Namespace, rsc: Resource):
        """
        Args:
            cfg (Namespace):  config
            rsc (Resource):  Resource object
        """
        super().__init__()
        self.cfg = cfg
        self.rsc = rsc
#        self.embedder = Embedder(cfg, rsc)
        self.embedder = Embedder1(cfg, rsc)

    def forward(self, *inputs):
        raise NotImplementedError

    def save(self, path: str):
        """
        모델을 저장하는 메소드
        Args:
            path (str):  경로
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """
        저장된 모델을 로드하는 메소드
        Args:
            path (str):  경로
        """
        if torch.cuda.is_available():
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.cuda()
            
class Embedder1(nn.Module):
    """
    embedder class
    """
    def __init__(self, cfg: Namespace, rsc: Resource):
        """
        Args:
            cfg:  config
            rsc:  Resource object
        """
        super().__init__()
        self.cfg = cfg
        self.rsc = rsc
        self.embedding = nn.Embedding(len(rsc.vocab_in), cfg.embed_dim)

    def forward(self, inputs):    # pylint: disable=arguments-differ
        """
        임베딩을 생성하는 메소드
        Args:
            inputs:  contexts of batch size
        Returns:
            embedding
        """
        embeds = self.embedding(inputs)
#        embeds_pretrained = load_embeds_fasttext(inputs)
#        embeds_final = torch.cat((embeds_pretrained, embeds), 1)
        return embeds


class LSTM_Model(PosModel):
    
    def __init__(self, cfg: Namespace, rsc: Resource):
        super().__init__(cfg, rsc)
        self.hidden_size = 500
        self.num_layers = 4
        self.rnn=torch.nn.LSTM(
            input_size= cfg.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.out1=torch.nn.Linear(in_features=self.hidden_size,out_features=len(rsc.vocab_out))
        self.out2=torch.nn.Linear(in_features=self.hidden_size,out_features=4)
        self.word_embeddings = nn.Embedding(len(rsc.vocab_in), cfg.embed_dim, padding_idx=0, scale_grad_by_freq=True)


    def forward(self, contexts, input_lengths, evaluate_mode=False):
        
        if evaluate_mode == False:
            self.hidden = self.init_hidden(input_lengths.size(0))
        else:
            self.hidden = self.init_hidden(1)
            
        embeds = self.word_embeddings(contexts)       
        
        total_length = contexts.size(1)
        pack = nn_utils.rnn.pack_padded_sequence(embeds, input_lengths, batch_first = True)
        self.rnn.flatten_parameters()
        output,self.hidden=self.rnn(pack, self.hidden) #或者embeds.view(1, -1, cfg.embed_dim)
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        
        unpacked = unpacked[:, :, :self.hidden_size] + unpacked[:, : ,self.hidden_size:]
        
        output_dropout = F.dropout(unpacked)
        
        tag_out = self.out1(output_dropout)
        space_out = self.out2(output_dropout)

        return tag_out, space_out
    
    def init_hidden(self, batchsize=100):

        hidden_a = torch.empty(self.num_layers*2, batchsize,  self.hidden_size)
        hidden_a = nn.init.xavier_uniform_(hidden_a)

        hidden_b = torch.empty(self.num_layers*2, batchsize,  self.hidden_size)
        hidden_b = nn.init.xavier_uniform_(hidden_b)
       
        if torch.cuda.is_available():
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)
