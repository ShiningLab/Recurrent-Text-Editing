#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'ning.shi@learnable.ai'

import numpy as np

class Evaluate():

      def __init__(self, config, targets, predictions):

            self.config = config
            self.tars = targets
            self.preds = predictions
            self.size = len(targets)
            self.token_acc = self.get_token_acc()
            self.seq_acc = self.get_seq_acc()

      def check_token(self, tar, pred):
            min_len = min([len(tar), len(pred)])
            return np.float32(sum(np.equal(tar[:min_len], pred[:min_len]))/len(tar))

      def check_seq(self, tar, pred):
            min_len = min([len(tar), len(pred)])
            if sum(np.equal(tar[:min_len], pred[:min_len])) == len(tar):
                  return 1
            return 0

      def get_token_acc(self):
            a = 0
            for i in range(self.size):
                  tar = self.tars[i]
                  pred = self.preds[i]
                  a += self.check_token(tar, pred)
            return np.float32(a/self.size)

      def get_seq_acc(self):
            a = 0
            for i in range(self.size):
                  tar = self.tars[i]
                  pred = self.preds[i]
                  a += self.check_seq(tar, pred)
            return np.float32(a/self.size)