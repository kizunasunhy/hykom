# -*- coding: utf-8 -*-


"""
part-of-speech tagger
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import Namespace
import json
import logging
import re

import torch
import torch.nn.functional as F

from khaiii.resource.resource import Resource
from khaiii.train.dataset import PosSentTensor
from khaiii.train.models import CnnModel


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
        self.model = CnnModel(self.cfg, self.rsc)
        self.model.load('{}/model.state'.format(model_dir))
        self.model.eval()

    def tag_raw(self, raw_sent: str, enable_restore: bool = True) -> PosSentTensor:
        """
        part-of-speech tagging at raw sentence
        Args:
            raw_sent:  raw input sentence
        Returns:
            PosSentTensor object
        """
        pos_sent = PosSentTensor(raw_sent)
        _, contexts = pos_sent.to_tensor(self.cfg, self.rsc, False)
        if torch.cuda.is_available():
            contexts = contexts.cuda()
        outputs = self.model(contexts)
        _, predicts = F.softmax(outputs, dim=1).max(1)
        tags = [self.rsc.vocab_out[t.item()] for t in predicts]
        pos_sent.set_pos_result(tags, self.rsc.restore_dic if enable_restore else None)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            raw_nospc = re.sub(r'\s+', '', raw_sent)
            for idx, (tag, pred) in enumerate(zip(tags, predicts)):
                logging.debug('[%2d]%s: %5s(%d)', idx, raw_nospc[idx], tag, pred.data[0])

        return pos_sent
