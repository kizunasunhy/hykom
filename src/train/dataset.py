# -*- coding: utf-8 -*-

"""
author: Sun Hongyang,
email: kizuna.sunhy@gmail.com
Reffered from khaiii, 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""

###########
# imports #
###########
from argparse import Namespace
import logging
import os
import random
from typing import List, TextIO, Tuple

import torch
from torch import LongTensor, Tensor    # pylint: disable=no-member, no-name-in-module
from torch.utils.data import Dataset
from tqdm import tqdm

from src.rsc.resource import PAD_CHR, Resource
from src.train.sentence import PosSentence, PosWord


#########
# types #
#########
class PosSentTensor(PosSentence):
    """
    tensor transformable sentence
    """
    def __init__(self, raw_sent: str = ''):
        super().__init__(raw_sent)
        if raw_sent:
            self.init_pos_tags()

    def __len__(self):
        # RNN에서 길이가 긴 것으로 정렬하기위한 문장의 길이 = 음절 갯수 + 문장 경계 + 어절 경계
        if self.words:
            return sum([len(w) for w in self.words]) + len(self.words) + 1
        if self.pos_tagged_words:
            return sum([len(w.raw) for w in self.pos_tagged_words]) + len(self.pos_tagged_words) + 1
        return 0

    def make_labels(self, with_spc: bool, rsc: Resource) -> List[str]:
        """
        각 음절별로 출력 레이블(태그)를 생성한다.
        Args:
            with_spc:  공백(어절 경계) 포함 여부
        Returns:
            레이블 리스트
        """
        if not with_spc:
            # 문장 경계, 어절 경계 등 가상 음절을 제외하고 순수한 음절들의 레이블
            labels = [tag for pos_word in self.pos_tagged_words for tag in pos_word.tags]
            label_nums = [rsc.vocab_out[l] for l in labels]
            return label_nums
        labels = [PAD_CHR, ]    # 문장 시작
        for pos_word in self.pos_tagged_words:
            if len(labels) > 1:
                labels.append(PAD_CHR)    # 어절 경계
            labels.extend(pos_word.tags)
        labels.append(PAD_CHR)    # 문장 종료
        return labels
    
    def make_space_labels(self, with_spc: bool, rsc: Resource) -> List[str]:
         if not with_spc:
            label = []
            for word in self.words:
                if len(word) == 1:
                    label.append(2)
                    continue
                else:
                    for a in range(len(word)-1):
                        label.append(1)
                    label.append(2)
                    
            try:
                label[len(label)-1] = 3
            except IndexError:
                print(self.words)
                
            return label   

    def make_contexts(self, spc_dropout: float) -> List[str]:

        contexts = []
        for wrd_idx, word in enumerate(self.words):
            for chr_idx, char in enumerate(word):
                contexts.append([char, ])
        
        return contexts
    

    def to_tensor(self, cfg: Namespace, rsc: Resource, is_train: bool) -> Tuple[Tensor, Tensor]:
        """
        문장 내에 포함된 전체 음절들과 태그를 모델의 forward 메소드에 넣을 수 있는 텐서로 변환한다.
        Args:
            cfg:  config
            rsc:  Resource object
            is_train:  whether is train or not
        Returns:
            labels tensor
            contexts tensor
        """
        # 차원: [문장내 음절 갯수, ]

        labels_tensor = LongTensor(self.make_labels(False, rsc))
        # 차원: [문장내 음절 갯수 x context 크기]
        space_labels_tensor = LongTensor(self.make_space_labels(False, rsc))
        spc_dropout = cfg.spc_dropout if is_train else 0.0
        context_nums = [[rsc.vocab_in[c] for c in context] \
                        for context in self.make_contexts(spc_dropout)]
        contexts_tensor = LongTensor(context_nums)       
        contexts_tensor = contexts_tensor.squeeze(1)
#        print(contexts_tensor) #应该是Tensor([一句话内的字的个数])

#        contexts_tensor = torch.cat((contexts_tensor, contexts_tensor), 0)
    
        return labels_tensor, space_labels_tensor, contexts_tensor
    

    def make_chars(self) -> List[str]:
        """
        문장 내 포함된 음절들을 만든다. 문장 경계 및 어절 경계를 포함한다.
        Returns:
            음절의 리스트
        """
        chars = ['<s>', ]    # 문장 시작
        for word in self.words:
            if len(chars) > 1:
                chars.append('<w>')    # 어절 경계
            chars.extend(word)
        chars.append('</s>')    # 문장 종료
        return chars


class PosDataset(Dataset):
    """
    part-of-speech tag dataset
    """
    def __init__(self, cfg: Namespace, restore_dic: dict, rsc: Resource, fin: TextIO):
        """
        Args:
            cfg:  config
            restore_dic:  restore dictionary
            fin:  input file
        """
        self.cfg = cfg
        self.rsc = rsc
        self.fin = fin
        self.sents = []
        self.sent_idx = -1
        self._load(restore_dic)
    
    def __getitem__(self, idx):

        return self.sents[idx].to_tensor(self.cfg, self.rsc, is_train = True)
    
    def __len__(self):
        return len(self.sents)

    def __str__(self):
        return '<PosDataset: file: {}, sents: {}, sent_idx: {}>'.format(
            os.path.basename(self.fin.name), len(self.sents), self.sent_idx)

    def _load(self, restore_dic: dict):
        """
        load data file
        Args:
            restore_dic:  restore dictionary
        """
        sent = PosSentTensor()
        lines = self.fin.readlines()
        for line in tqdm(lines, os.path.basename(self.fin.name), len(lines), mininterval=1,
                         ncols=100):
            line = line.rstrip('\r\n')
#            print(line) #就是原corpus数据中的每一行，包括空的一行（句子与句子之间的区分） 如 집안의  I-NNG I-NNG I-JKG
            if not line:
                if sent and sent.pos_tagged_words:
                    sent.set_raw_by_words()
                    self.sents.append(sent)
                sent = PosSentTensor()
                continue
            raw, tags = line.split('\t')
#            print(raw)  #line的左半部分，一句话中的单词
#            print(tags) #line的右半部分，单词对应的标签
            pos_word = PosWord(raw)
#            print(pos_word) #如 집안의  집/O 안/O 의/O
            pos_word.set_pos_result(tags.split(), restore_dic)
            sent.pos_tagged_words.append(pos_word)
#        print(self.sents) #PosSentTensor Object
        logging.info('%s: %d sentences', os.path.basename(self.fin.name), len(self.sents))

    def __iter__(self):
        self.sent_idx = -1
        random.shuffle(self.sents)
        return self

    def __next__(self):
        self.sent_idx += 1
        if self.sent_idx >= len(self.sents):
            raise StopIteration()
        return self.sents[self.sent_idx]

    
    