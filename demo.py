# -*- coding: utf-8 -*-

"""
author: Sun Hongyang,
email: kizuna.sunhy@gmail.com
"""

###########
# imports #
###########
from argparse import ArgumentParser, Namespace
import json
import logging
import re
import os
import sys
import time

import torch
import torch.nn.functional as F

from rsc.resource import Resource
from src.train.dataset import PosSentTensor
from src.train.models import LSTM_Model
from torch import FloatTensor, LongTensor, Tensor


#########
# types #
#########
class PosTagger:
    """
    part-of-speech tagger
    """
    def __init__(self, model_dir: str):
        """
        Args:
            model_dir:  model dir
        """
            
        cfg_dict = json.load(open('{}/config.json'.format(model_dir), 'r', encoding='UTF-8'))
        self.cfg = Namespace()
        for key, val in cfg_dict.items():
            setattr(self.cfg, key, val)
        self.rsc = Resource(self.cfg)
        self.model = LSTM_Model(self.cfg, self.rsc)
        self.model.load('{}/model.state'.format(model_dir))
        self.model.eval()
        

    def tag_raw(self, raw_sent: str, space:bool, enable_restore: bool = True) -> PosSentTensor:
        """
        part-of-speech tagging at raw sentence
        Args:
            raw_sent:  raw input sentence
        Returns:
            PosSentTensor object
        """
        seed = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
#        sent_vec = FloatTensor(array)
        pos_sent = PosSentTensor(raw_sent)
#        pos_sent = PosSentTensor('안녕')
        _, _, contexts = pos_sent.to_tensor(self.cfg, self.rsc, False)

        context_lengths = len(contexts)
            
        contexts = contexts.resize_(1, context_lengths)
              
        if torch.cuda.is_available():
            contexts = contexts.cuda()

        outputs, space_outputs = self.model(contexts, [context_lengths], True)
        
        outputs.contiguous()
        outputs = outputs.view(context_lengths, -1)
        space_outputs.contiguous()
        space_outputs = space_outputs.view(context_lengths, -1)
       

        _, predicts = F.softmax(outputs, dim=1).max(1)
        _, space_predicts = F.softmax(space_outputs, dim=1).max(1)

        tags = [self.rsc.vocab_out[t.item()] for t in predicts]
        pos_sent.set_pos_result(tags, self.rsc.restore_dic if enable_restore else None)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            raw_nospc = re.sub(r'\s+', '', raw_sent)
            for idx, (tag, pred) in enumerate(zip(tags, predicts)):
                logging.debug('[%2d]%s: %5s(%d)', idx, raw_nospc[idx], tag, pred.data[0])
        if space == False:
            return pos_sent, space_predicts
        else:
            return self.tag_space(raw_sent, space_predicts, tags)
    
    def tag_space(self, raw_sent:str, space_result:list, tags:list, enable_restore: bool = True):
        sentence_space = []
        for ind in range(len(raw_sent)):
            sentence_space.append(raw_sent[ind])
            if space_result[ind] == 2:
                sentence_space.append(' ')
            if space_result[ind] == 3:
                sentence_space.append('\n')

        sentence_with_space = ''.join(sentence_space)
        print(sentence_with_space)
        pos_sent_space = PosSentTensor(sentence_with_space)
        pos_sent_space.set_pos_result(tags, self.rsc.restore_dic if enable_restore else None)
        
        return pos_sent_space


#############
# functions #
#############
def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """    
    tgr = PosTagger(args.model_dir)
    #embeds_model = fasttext.load_model('model_drama.bin')  
    
        
    for line_num, line in enumerate(sys.stdin, start=1):

        if line_num % 100000 == 0:
            logging.info('%d00k-th line..', (line_num // 100000))
        line = line.rstrip('\r\n')
        line = line.replace(' ', '')
        line = line.replace('\t', '')
        if not line:
            print()
            continue

        if args.without_space == True:
            pos_sent, space_result = tgr.tag_raw(line, False) 
            sentence_space = []
            for ind in range(len(line)):
                sentence_space.append(line[ind])
                if space_result[ind] == 1:
                    sentence_space.append(' ')   
            sentence_with_space = ''.join(sentence_space)
            print(sentence_with_space)
            for pos_word in pos_sent.pos_tagged_words:
                print(pos_word.raw, end='\t')
                print(' + '.join([str(m) for m in pos_word.pos_tagged_morphs]))
            print()
        else:
            pos_sent_space = tgr.tag_raw(line, True)
            for pos_word in pos_sent_space.pos_tagged_words:
                print(pos_word.raw, end='\t')
                print(' + '.join([str(m) for m in pos_word.pos_tagged_morphs]))
            print()


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='command line part-of-speech tagger demo')
    parser.add_argument('-m', '--model-dir', help='model dir', metavar='DIR', default='./logdir/corpus.cut2.sdo0.0.emb100.lr0.001.lrd0.6.bs250')
    parser.add_argument('--without_space', help='output the morpheme result with or without space decomposition', default=False, action='store_true')
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--gpu_num', help='GPU number to use', metavar='INT', type=int, default=0)
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    if args.input:
        sys.stdin = open(args.input, 'r', encoding='UTF-8')
    if args.output:
        sys.stdout = open(args.output, 'w', encoding='UTF-8')
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)

if __name__ == '__main__':
    main()