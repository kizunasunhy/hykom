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
import copy
from datetime import datetime, timedelta
import json
import logging
import os
import pathlib
import pprint
from typing import List, Tuple
from collections import Counter
import time

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from src.train.dataset import PosDataset
from src.train.evaluator import Evaluator
from src.train.models import LSTM_Model
from src.rsc.resource import Resource

    
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

#############
# functions #
#############
class Trainer:
    """
    trainer class
    """
    def __init__(self, cfg: Namespace):
        """
        Args:
            cfg:  config
        """
        self.cfg = cfg
        self.cnt = Counter()
        setattr(cfg, 'model_id', self.model_id(cfg))
        setattr(cfg, 'out_dir', '{}/{}'.format(cfg.logdir, cfg.model_id))
        self.rsc = Resource(cfg)

        self.model = LSTM_Model(cfg, self.rsc)
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), cfg.learning_rate)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.learning_rate, momentum=0.9)    
        
        self.criterion = nn.CrossEntropyLoss()
        self.evaler = Evaluator()
        self._load_dataset()
        if 'epoch' not in cfg.__dict__:
            setattr(cfg, 'epoch', 0)
            setattr(cfg, 'best_epoch', 0)
        self.log_file = None    # tab separated log file
        self.sum_wrt = None    # tensorboard summary writer
        self.loss_trains = []
        self.loss_devs = []
        self.acc_chars = []
        self.acc_words = []
        self.f_scores = []
        self.learning_rates = []
        if self.cfg.fp16:
            from apex import amp    
            self.amp = amp
    @classmethod
    def model_id(cls, cfg: Namespace) -> str:
        """
        get model ID
        Args:
            cfg:  config
        Returns:
            model ID
        """
        model_cfgs = [
            os.path.basename(cfg.in_pfx),
            'cut{}'.format(cfg.cutoff),
            'sdo{}'.format(cfg.spc_dropout),
            'emb{}'.format(cfg.embed_dim),
            'lr{}'.format(cfg.learning_rate),
            'lrd{}'.format(cfg.lr_decay),
            'bs{}'.format(cfg.batch_size),
        ]
        return '.'.join(model_cfgs)

    def _load_dataset(self):
        """
        load training dataset
        """
        dataset_dev_path = '{}.test'.format(self.cfg.in_pfx)
        self.dataset_dev = PosDataset(self.cfg, self.rsc.restore_dic, self.rsc,
                                      open(dataset_dev_path, 'r', encoding='UTF-8'))
        dataset_test_path = '{}.test'.format(self.cfg.in_pfx)
        self.dataset_test = PosDataset(self.cfg, self.rsc.restore_dic, self.rsc,
                                       open(dataset_test_path, 'r', encoding='UTF-8'))
        dataset_train_path = '{}.test'.format(self.cfg.in_pfx)
        self.dataset_train = PosDataset(self.cfg, self.rsc.restore_dic, self.rsc,
                                        open(dataset_train_path, 'r', encoding='UTF-8'))
        
    @classmethod
    def _dt_str(cls, dt_obj: datetime) -> str:
        """
        string formatting for datetime object
        Args:
            dt_obj:  datetime object
        Returns:
            string
        """
        return dt_obj.strftime('%m/%d %H:%M:%S')

    @classmethod
    def _elapsed(cls, td_obj: timedelta) -> str:
        """
        string formatting for timedelta object
        Args:
            td_obj:  timedelta object
        Returns:
            string
        """
        seconds = td_obj.seconds
        if td_obj.days > 0:
            seconds += td_obj.days * 24 * 3600
        hours = seconds // 3600
        seconds -= hours * 3600
        minutes = seconds // 60
        seconds -= minutes * 60
        return '{}:{:02d}:{:02d}'.format(hours, minutes, seconds)

    def _restore_prev_train(self):
        """
        기존에 학습하다 중지한 경우 그 이후부터 계속해서 학습할 수 있도록 이전 상태를 복원한다.
        """
        out_path = pathlib.Path(self.cfg.out_dir)
        cfg_path = pathlib.Path('{}/config.json'.format(self.cfg.out_dir))
        if not out_path.is_dir() or not cfg_path.is_file():
            return False
        logging.info('==== continue training: %s ====', self.cfg.model_id)
        cfg = json.load(open(cfg_path, 'r', encoding='UTF-8'))
        for key, val in cfg.items():
            setattr(self.cfg, key, val)
        self._revert_to_best(False)

        f_score_best = 0.0
        best_idx = -1
        for idx, line in enumerate(open('{}/log.tsv'.format(self.cfg.out_dir))):
            line = line.rstrip('\r\n')
            if not line:
                continue
            (epoch, loss_train, loss_dev, acc_char, acc_word, f_score, learning_rate) = \
                    line.split('\t')
            self.cfg.epoch = int(epoch) + 1
            self.cfg.best_epoch = self.cfg.epoch
            self.loss_trains.append(float(loss_train))
            self.loss_devs.append(float(loss_dev))
            self.acc_chars.append(float(acc_char))
            self.acc_words.append(float(acc_word))
            self.f_scores.append(float(f_score))
            self.learning_rates.append(float(learning_rate))
            if float(f_score) > f_score_best:
                f_score_best = float(f_score)
                best_idx = idx
        logging.info('---- [%d] los(trn/dev): %.4f / %.4f, acc(chr/wrd): %.4f / %.4f, ' \
                     'f-score: %.4f, lr: %.8f ----', self.cfg.epoch,
                     self.loss_trains[best_idx], self.loss_devs[best_idx], self.acc_chars[best_idx],
                     self.acc_words[best_idx], self.f_scores[best_idx], self.learning_rates[-1])
        return True
    
    def train(self):
        """
        train model with dataset
        """
        continue_train = self._restore_prev_train()
        logging.info('config: %s', pprint.pformat(self.cfg.__dict__))

        train_begin = datetime.now()
              
        logging.info('{{{{ training begin: %s {{{{', self._dt_str(train_begin))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        #print("n_gpu=", n_gpu)
        logger.info("device: {} n_gpu: {}".format(
        device, n_gpu))
        if torch.cuda.is_available():
            self.model.to(device)
            
        #torch.distributed.init_process_group(backend='nccl', world_size=2, init_method='...')
        #self.model = torch.nn.DataParallel(self.model)
        
        if continue_train != True:
            if self.cfg.fp16:
                #from apex import amp
                self.model, self.optimizer = self.amp.initialize(self.model, self.optimizer, opt_level="O1")
            
        pathlib.Path(self.cfg.out_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = open('{}/log.tsv'.format(self.cfg.out_dir), 'at')
        self.sum_wrt = SummaryWriter(self.cfg.out_dir)
        patience = self.cfg.patience
        
        for _ in range(10):
            is_best = self._train_epoch()            
            
            if is_best:
                
                self.cfg.learning_rate *= self.cfg.lr_decay        
                self._load_optim('{}/optim.state'.format(self.cfg.out_dir), self.cfg.learning_rate)
            
                patience = self.cfg.patience
                continue
            if patience <= 7:
                break
            self._revert_to_best(True)
            patience -= 1
            logging.info('==== revert to EPOCH[%d], f-score: %.4f, patience: %d ====',
                         self.cfg.best_epoch, max(self.f_scores), patience)
        
        train_end = datetime.now()
        train_elapsed = self._elapsed(train_end - train_begin)
        logging.info('}}}} training end: %s, elapsed: %s, epoch: %s }}}}',
                     self._dt_str(train_end), train_elapsed, self.cfg.epoch)
        
        eval_begin = datetime.now()
        avg_loss, acc_char, acc_word, f_score, space_f_score, _ = self.evaluate3(False)
        eval_end = datetime.now()
        eval_elapsed = self._elapsed(eval_end - eval_begin)
        logging.info('==== test loss: %.4f, char acc: %.4f, word acc: %.4f, f-score: %.4f, space f-score: %.4f ====',
                     avg_loss, acc_char, acc_word, f_score, space_f_score)
        print('eval time', eval_elapsed)

    def _revert_to_best(self, is_decay_lr: bool):
        """
        이전 best 모델로 되돌린다.
        Args:
            is_decay_lr:  whether multiply decay factor or not
        """
        self.model.load('{}/model.state'.format(self.cfg.out_dir))
        '''
        params=self.model.state_dict() #获得模型的原始状态以及参数。
        for k,v in params.items():
            print(k)
        '''
        if is_decay_lr:
            self.cfg.learning_rate *= self.cfg.lr_decay        
        self._load_optim('{}/optim.state'.format(self.cfg.out_dir), self.cfg.learning_rate, )
    
    def my_collate(self, batch):
        
        batch_size = self.cfg.batch_size

        X_lengths = [len(x[2]) for x in batch]
        #print(X_lengths)
        longest_sent = max(X_lengths)
        pad_token = 0
        padded_train_contexts = torch.ones((len(batch), longest_sent)).long() * pad_token
        for i, x_len in enumerate(X_lengths):
            sequence = batch[i][2]
            padded_train_contexts[i, 0:x_len] = sequence[:x_len]

        context_lengths = torch.LongTensor([torch.max(padded_train_contexts[i, :].data.nonzero())+1 for i in range(padded_train_contexts.size(0))])
        context_lengths, sorted_idx = context_lengths.sort(0, descending=True)
        padded_train_contexts = padded_train_contexts[sorted_idx]
                
        Y_lengths = [len(x[1]) for x in batch]
        padded_space_labels = torch.ones((len(batch), longest_sent)).long() * pad_token
        for i, y_len in enumerate(Y_lengths):
            sequence = batch[i][1]
            padded_space_labels[i, 0:y_len] = sequence[:y_len]         
        padded_space_labels = padded_space_labels[sorted_idx]
                
        Z_lengths = [len(x[0]) for x in batch]
        padded_labels = torch.ones((len(batch), longest_sent)).long() * pad_token
        for i, z_len in enumerate(Z_lengths):
            sequence = batch[i][0]
            padded_labels[i, 0:z_len] = sequence[:z_len]
        padded_labels = padded_labels[sorted_idx]
            
        return padded_train_contexts, padded_space_labels, padded_labels, context_lengths

    def _train_epoch(self) -> bool:
        
        loss_trains = []
        #count = 0
        #train_contexts_tmp = []                        
        #batch_size = self.cfg.batch_size
        #embeds_model = fasttext.load_model('model_drama.bin')
        dsloader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.cfg.batch_size, 
                                               collate_fn=self.my_collate, num_workers=2, pin_memory=False)       
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        
        for padded_train_contexts, padded_space_labels, padded_labels, context_lengths in tqdm(dsloader, 'EPOCH[{}]'.format(self.cfg.epoch), mininterval=1, ncols=100):                               
                        
            if torch.cuda.is_available():
                padded_train_contexts = padded_train_contexts.cuda() 
                padded_space_labels = padded_space_labels.cuda()
                padded_labels = padded_labels.cuda()
                context_lengths = context_lengths.cuda()

            self.model.train()
            
            train_outputs, train_space_outputs = self.model(padded_train_contexts, 
                                    context_lengths)
            
            
            train_outputs.requires_grad_()
            train_space_outputs.requires_grad_()
            
            padded_labels = padded_labels.view(-1)
            padded_space_labels = padded_space_labels.view(-1)
            
            train_outputs = F.log_softmax(train_outputs, dim=2)
            train_space_outputs = F.log_softmax(train_space_outputs, dim=2)
            
            train_outputs = train_outputs.view(-1, len(self.rsc.vocab_out))
            train_space_outputs = train_space_outputs.view(-1, 4)
            
            mask = (padded_labels > 0).float()
            mask1 = (padded_space_labels > 0).float()
            assert all(mask) == all(mask1)
            original_len = int(torch.sum(mask))
            train_outputs = train_outputs[range(train_outputs.shape[0]), padded_labels] * mask
            train_space_outputs = train_space_outputs[range(train_space_outputs.shape[0]), padded_space_labels] * mask

            loss_train = -torch.sum(train_outputs) / original_len
            loss_space_train = -torch.sum(train_space_outputs) / original_len
            total_loss = loss_train + loss_space_train
            #print(loss_train.item())
            
            
            loss_trains.append(loss_train.item())
            self.optimizer.zero_grad()
            if self.cfg.fp16:
                with self.amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()

            
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            
        avg_loss_dev, acc_char, acc_word, f_score, space_f_score, avg_space_loss_dev = self.evaluate(True)
        is_best = self._check_epoch(loss_trains, avg_loss_dev, acc_char, acc_word, f_score)
        print('Average space loss is: %.4f, space f1 score is: %.4f' % (avg_space_loss_dev, space_f_score))
        self.cfg.epoch += 1

        return is_best

    def _check_epoch(self, loss_trains: List[float], avg_loss_dev: float, acc_char: float,
                     acc_word: float, f_score: float) -> bool:
        """
        매 epoch마다 수행하는 체크
        Args:
            loss_trains:   train 코퍼스에서 각 배치별 loss 리스트
            avg_loss_dev:  dev 코퍼스 문장 별 평균 loss
            acc_char:  음절 정확도
            acc_word:  어절 정확도
            f_score:  f-score
        Returns:
            현재 epoch이 best 성능을 나타냈는 지 여부
        """
        avg_loss_train = sum(loss_trains) / len(loss_trains)
        loss_trains.clear()
        self.loss_trains.append(avg_loss_train)
        self.loss_devs.append(avg_loss_dev)
        self.acc_chars.append(acc_char)
        self.acc_words.append(acc_word)
        self.f_scores.append(f_score)
        self.learning_rates.append(self.cfg.learning_rate)
        is_best = self._is_best()
        is_best_str = 'BEST' if is_best else '< {:.4f}'.format(max(self.f_scores))
        logging.info('[Los trn]  [Los dev]  [Acc chr]  [Acc wrd]  [F-score]           [LR]')
        logging.info('{:9.4f}  {:9.4f}  {:9.4f}  {:9.4f}  {:9.4f} {:8}  {:.8f}' \
                .format(avg_loss_train, avg_loss_dev, acc_char, acc_word, f_score, is_best_str,
                        self.cfg.learning_rate))
        print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(self.cfg.epoch, avg_loss_train, avg_loss_dev,
                                                  acc_char, acc_word, f_score,
                                                  self.cfg.learning_rate), file=self.log_file)
        self.log_file.flush()
        self.sum_wrt.add_scalar('loss-train', avg_loss_train, self.cfg.epoch)
        self.sum_wrt.add_scalar('loss-dev', avg_loss_dev, self.cfg.epoch)
        self.sum_wrt.add_scalar('acc-char', acc_char, self.cfg.epoch)
        self.sum_wrt.add_scalar('acc-word', acc_word, self.cfg.epoch)
        self.sum_wrt.add_scalar('f-score', f_score, self.cfg.epoch)
        self.sum_wrt.add_scalar('learning-rate', self.cfg.learning_rate, self.cfg.epoch)
        return is_best

    def _is_best(self) -> bool:
        """
        이번 epoch에 가장 좋은 성능을 냈는 지 확인하고 그럴 경우 현재 상태를 저장한다.
        Returns:
            마지막 f-score의 best 여부
        """
        if len(self.f_scores) > 1 and max(self.f_scores[:-1]) >= self.f_scores[-1]:
            return False
        # this epoch hits new max value
        self.cfg.best_epoch = self.cfg.epoch
        self.model.save('{}/model.state'.format(self.cfg.out_dir))
        #torch.save(self.model, '{}/model.pt'.format(self.cfg.out_dir))
        self._save_optim('{}/optim.state'.format(self.cfg.out_dir))
        with open('{}/config.json'.format(self.cfg.out_dir), 'w', encoding='UTF-8') as fout:
            json.dump(vars(self.cfg), fout, indent=2, sort_keys=True)
        return True

    def _save_optim(self, path: str):
        """
        save optimizer parameters
        Args:
            path:  path
        """
        torch.save(self.optimizer.state_dict(), path)

    def _load_optim(self, path: str, learning_rate: float):
        """
        load optimizer parameters
        Args:
            path:  path
            learning_rate:  learning rate
        """
        if torch.cuda.is_available():
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        if self.cfg.fp16:
            self.model, self.optimizer = self.amp.initialize(self.model, self.optimizer, opt_level="O1")
        self.optimizer.load_state_dict(state_dict)
        self.optimizer.param_groups[0]['lr'] = learning_rate

    
    def evaluate(self, is_dev: bool) -> Tuple[float, float, float, float]:

        dataset = self.dataset_dev if is_dev else self.dataset_test
        self.model.eval()
        losses = []
        losses_space = []
        count = 0
        contexts_tmp = []
        batch_size = 400

        
        for sent in dataset:
            # 만약 spc_dropout이 1.0 이상이면 공백을 전혀 쓰지 않는 것이므로 평가 시에도 적용한다.
            labels, space_labels, contexts = sent.to_tensor(self.cfg, self.rsc, self.cfg.spc_dropout >= 1.0)
            contexts_tmp.append((labels, space_labels, contexts, sent))
            count += 1
            if count < batch_size:
                continue 
            
            X_lengths = [len(x[2]) for x in contexts_tmp]
            longest_sent = max(X_lengths)
            pad_token = 0
            padded_contexts = torch.ones((batch_size, longest_sent)).long() * pad_token
            for i, x_len in enumerate(X_lengths):
                sequence = contexts_tmp[i][2]
                padded_contexts[i, 0:x_len] = sequence[:x_len]
            
            context_lengths = torch.LongTensor([torch.max(padded_contexts[i, :].data.nonzero())+1 for i in range(padded_contexts.size(0))])
            context_lengths, sorted_idx = context_lengths.sort(0, descending=True)
            padded_contexts = padded_contexts[sorted_idx]
            
            Y_lengths = [len(x[1]) for x in contexts_tmp]
            padded_space_labels = torch.ones((batch_size, longest_sent)).long() * pad_token
            for i, y_len in enumerate(Y_lengths):
                sequence = contexts_tmp[i][1]
                padded_space_labels[i, 0:y_len] = sequence[:y_len]         
            padded_space_labels = padded_space_labels[sorted_idx]
                
            Z_lengths = [len(x[0]) for x in contexts_tmp]
            padded_labels = torch.ones((batch_size, longest_sent)).long() * pad_token
            for i, z_len in enumerate(Z_lengths):
                sequence = contexts_tmp[i][0]
                padded_labels[i, 0:z_len] = sequence[:z_len]
            padded_labels = padded_labels[sorted_idx]
                    
            contexts_tmp1 = [0]*batch_size
            contexts_tmp2 = [0]*batch_size
            for a in range(batch_size):
                contexts_tmp1[a] = contexts_tmp[sorted_idx[a]][3]
                contexts_tmp2[a] = contexts_tmp[sorted_idx[a]][1]
                
            count = 0           
                
            if torch.cuda.is_available():
                padded_contexts = padded_contexts.cuda() 
                padded_space_labels = padded_space_labels.cuda()
                padded_labels = padded_labels.cuda()
                context_lengths = context_lengths.cuda()

            outputs, outputs_space = self.model(padded_contexts, context_lengths)
              
            padded_labels = padded_labels.view(-1)
            padded_space_labels = padded_space_labels.view(-1)

            outputs = outputs.view(-1, len(self.rsc.vocab_out))
            outputs_space = outputs_space.view(-1, 4)
            
            loss_eval = self.criterion(outputs, padded_labels)
            loss_space_eval = self.criterion(outputs_space, padded_space_labels)
            losses.append(loss_eval.item())
            losses_space.append(loss_space_eval.item())
      
            _, predicts = F.softmax(outputs, dim=1).max(1)
            _, predicts_space = F.softmax(outputs_space, dim=1).max(1)
            #print(predicts.size(), predicts_space)
            
            for n in range(batch_size):
                pred_sent = copy.deepcopy(contexts_tmp1[n])
                pred_tags = [self.rsc.vocab_out[t.item()] for t in predicts[(n*longest_sent):(n*longest_sent+context_lengths[n])]]    
                pred_sent.set_pos_result(pred_tags, self.rsc.restore_dic)
                space = contexts_tmp2[n]
                self.evaler.count(contexts_tmp1[n], pred_sent)
                self.evaler.count_space(space, predicts_space[n*longest_sent:(n*longest_sent+context_lengths[n])])
                
            contexts_tmp = []    
        avg_loss = sum(losses) / len(losses)
        avg_space_loss = sum(losses_space) / len(losses_space)
        return (avg_loss, ) + self.evaler.evaluate() + (avg_space_loss, )
    