# -*- coding: utf-8 -*-
"""
author: Sun Hongyang,
email: kizuna.sunhy@gmail.com
Reffered from khaiii, 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""

###########
# imports #
###########
from argparse import ArgumentParser, Namespace
import json
import re
import os
import sys

import torch
import torch.nn.functional as F
import time

from src.rsc.resource import Resource
from src.train.dataset import PosSentTensor
from src.train.models import LSTM_Model
from torch import FloatTensor
from flask import Flask, request, Response

#app = Flask(__name__)

    
#########
# class #
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
        
        if torch.cuda.is_available():
            self.model.load('{}/model.state'.format(model_dir))
        else:
            device = torch.device('cpu')        
            self.model.load_state_dict(torch.load('{}/model.state'.format(model_dir), map_location=device))
            
        #self.model.load('{}/model.state'.format(model_dir))
        self.model.eval()

    def tag_raw(self, raw_sent: str, sent_original:str, enable_restore: bool = True) -> PosSentTensor:
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
        
        #print(list(raw_sent))
        pos_sent = PosSentTensor(raw_sent)

        _, _, contexts = pos_sent.to_tensor(self.cfg, self.rsc, False)

        context_lengths = len(contexts)
        contexts = contexts.resize_(1, context_lengths)
        if torch.cuda.is_available():
            contexts = contexts.cuda()

        outputs, space_outputs = self.model(contexts, [context_lengths], True)
        #outputs, space_outputs = self.model_trt(contexts, [context_lengths], True)
        
        outputs.contiguous()
        outputs = outputs.view(context_lengths, -1)
        space_outputs.contiguous()
        space_outputs = space_outputs.view(context_lengths, -1)

        _, predicts = F.softmax(outputs, dim=1).max(1)
        _, space_predicts = F.softmax(space_outputs, dim=1).max(1)

        tags = [self.rsc.vocab_out[t.item()] for t in predicts]
        #pos_sent.set_pos_result(tags, self.rsc.restore_dic if enable_restore else None)
        
        use_original_sentence = False
        
        if use_original_sentence is True:
            pos_sent_space, sentence_with_space = self.tag_space2(raw_sent, sent_original, space_predicts, tags)
        else:
            pos_sent_space, sentence_with_space = self.tag_space1(raw_sent, sent_original, space_predicts, tags)

        return pos_sent_space, sentence_with_space
  
    
    def tag_space1(self, raw_sent:str, sent_original:str, space_result:list, tags:list, enable_restore: bool = True):
        sentence_space = []
        for ind in range(len(raw_sent)):
            sentence_space.append(raw_sent[ind])
            
            try:
                if space_result[ind] == 2:
                    sentence_space.append(' ')
            except IndexError:
                print("sentence: %s has an index error, index is %s" % (raw_sent, ind))
        #print(sentence_space) 
        sentence_with_space = ''.join(sentence_space)
        pos_sent_space = PosSentTensor(sentence_with_space)
        pos_sent_space.set_pos_result(tags, self.rsc.restore_dic if enable_restore else None)
        
        return pos_sent_space, sentence_with_space
    
    
    def tag_space2(self, raw_sent:str, sent_original:str, space_result:list, tags:list, enable_restore: bool = True):
        sentence_space = []
        for ind in range(len(raw_sent)):
            sentence_space.append(raw_sent[ind])
            try:
                if space_result[ind] == 2:
                    sentence_space.append(' ')
            except IndexError:
                print("sentence: %s has an index error, index is %s" % (raw_sent, ind))
    
        sentence_with_space = sent_original
        #sentence_with_space = ''.join(sentence_space)
        pos_sent_space = PosSentTensor(sentence_with_space)
        pos_sent_space.set_pos_result(tags, self.rsc.restore_dic if enable_restore else None)
        
        return pos_sent_space, sentence_with_space

#############
# functions #
#############

def create_app():
    # create and configure the app

    app = Flask(__name__)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
    
    model_dir = './logdir/corpus.cut2.sdo0.0.emb100.lr0.001.lrd0.6.bs250'
    tgr = PosTagger(model_dir)
    
    cfg_dict = json.load(open('{}/config.json'.format(model_dir), 'r', encoding='UTF-8'))
    cfg = Namespace()
    for key, val in cfg_dict.items():
        setattr(cfg, key, val)    
    rsc = Resource(cfg)
    print(rsc.users_dic)
    
    #return only nouns
    @app.route("/question1", methods=["POST"])
    def get_keywords():

        a=request.get_json()
        start_time = time.time()
        
        #print(a['text'])
        
        #line = a['text'].rstrip()
        line = a['text'].rstrip('\r\n')
        line = line.strip('"')
        line = line.replace(u'\u2003', '')
        line = line.replace(u'\u3000', '')
        line = line.replace('\t', '')
        line = line.replace('\n', '')
        line = line.replace('\\n', '')
        line_original = line
        line = line.replace(' ', '')
        
        pos_sent_space, sentence = tgr.tag_raw(line, line_original, True)
                
        keyword1 = []
        keyword2 = []  
        
        return_list = ['NNG', 'NNP', 'SL']           
        for pos_word in pos_sent_space.pos_tagged_words_idx:
            for morph in pos_word[0].pos_tagged_morphs:
                if morph.pos_tag in return_list:
                    keyword2.append(morph.morph)
                    
        keywords2 = list(set(keyword2)) 
        keywords2.sort(key=keyword2.index)
    
        json_dict={}
        json_dict['keywords']= keywords2
        json_dict['sentence']= sentence

        json_keywords=json.dumps(json_dict, ensure_ascii=False)    
        #print('It takes: ' + str(time.time() - start_time) + ' s for inference.')
    
        return json_keywords
    
    #return all morphemes with semantic significance
    @app.route("/question2", methods=["POST"])
    def get_morpheme3():

        a=request.get_json()
        start_time = time.time()
        
        #print(a['text'])
        
        #line = a['text'].rstrip()
        line = a['text'].rstrip('\r\n')
        line = line.strip('"')
        line = line.replace(u'\u2003', '')
        line = line.replace(u'\u3000', '')
        line = line.replace('\t', '')
        line = line.replace('\n', '')
        line = line.replace('\\n', '')
        line_original = line
        line = line.replace(' ', '')
            
        pos_sent_space, sentence = tgr.tag_raw(line, line_original, True)
                
        morphemes = []
        #pos_sent_space.index_list
        return_list = ['NNG', 'NNP', 'NNB', 'NR', 'SL', 'VV', 'VX', 'XSV', 'VA', 'MM', 'JX', 'XSN']
        for pos_word in pos_sent_space.pos_tagged_words_idx:
            for morph in pos_word[0].pos_tagged_morphs:
                if morph.pos_tag in return_list:
                    morphemes.append(morph.morph)
                    
        #print(morphemes)
        json_dict={}
        json_dict['morphemes']= morphemes

        json_morphemes=json.dumps(json_dict, ensure_ascii=False)    
        #print('It takes: ' + str(time.time() - start_time) + ' s for inference.')
    
        return Response(json_morphemes, mimetype='application/json')
    
    #return all the morphemes
    @app.route("/question3", methods=["POST"])
    def get_morpheme4():

        a=request.get_json()
        start_time = time.time()
        
        #line = a['text'].rstrip()
        line = a['text'].rstrip('\r\n')
        line = line.strip('"')
        line = line.replace(u'\u2003', '')
        line = line.replace(u'\u3000', '')
        line = line.replace('\t', '')
        line = line.replace('\n', '')
        line = line.replace('\\n', '')
        line_original = line
        line = line.replace(' ', '')
            
        pos_sent_space, sentence = tgr.tag_raw(line, line_original, True)
                
        morphemes = []
         
        for pos_word in pos_sent_space.pos_tagged_words:
            for m in pos_word.pos_tagged_morphs:
                morphemes.append(str(m))
        
        morpheme_return = ' '.join(morphemes)

        json_dict={}
        json_dict['morphemes']= morpheme_return

        json_morphemes=json.dumps(json_dict, ensure_ascii=False)    
        #print('It takes: ' + str(time.time() - start_time) + ' s for inference.')
    
        return Response(json_morphemes, mimetype='application/json')
    
    return app

app = create_app()

'''
if __name__ == "__main__":
    model_dir = './logdir/corpus.cut2.sdo0.0.emb100.lr0.001.lrd0.6.bs250'
    tgr = PosTagger(model_dir) 
    app.run(host='0.0.0.0',port=5050,debug=True)
'''