#!/usr/bin/env python
# -*- coding:utf-8 -*-

# dependency
# public
import numpy as np

class Evaluate():
    """a class to process evaluation"""
    def __init__(self, config, targets, predictions, idx2vocab_dict, train=False): 
        self.config = config
        self.idx2vocab_dict = idx2vocab_dict
        self.tars = targets
        self.preds = predictions
        self.size = len(targets)
        # token-level accuracy
        self.token_acc = self.get_token_acc()
        # sequence-level accuracy
        self.seq_acc = self.get_seq_acc()
        # main metric for early stopping
        self.key_metric = self.token_acc
        # generate an evaluation message
        self.eva_msg = 'Token Acc:{:.4f} Seq Acc:{:.4f}'.format(self.token_acc, self.seq_acc)
        if self.config.data_src in ['aor', 'aec'] and not train:
            # if hold equation 
            self.eq_acc = self.get_eq_acc()
            # main metric for early stopping
            self.key_metric = self.eq_acc
            # generate an evaluation message
            self.eva_msg += ' Equation Acc:{:.4f}'.format(self.eq_acc)

    def check_equation(self, tgt, pred): 
        # remove end symbol
        if self.config.method == 'e2e':
            # remove end symbol
            tgt = [t for t in tgt if t != self.config.end_idx]
            pred = [p for p in pred if p != self.config.end_idx]
        # e.g., ['3', '-', '3', '+', '9', '-', '3', '==', '6']
        tgt = [self.idx2vocab_dict[t] for t in tgt]
        # e.g., ['3', '-', '3', '+', '9', '+', '3', '==', '6']
        pred = [self.idx2vocab_dict[p] for p in pred]
        # e.g., ['3', '3', '9', '3', '6']
        tgt_nums = [t for t in tgt if t.isdigit()]
        # e.g., ['3', '3', '9', '3', '6']
        pred_nums = [p for p in pred if p.isdigit()]
        # eval('123') return 123
        # eval('1 2 3') raise error
        try:
            if tgt_nums == pred_nums and pred[-1].isdigit() and pred[-2] == '==':
                    right = int(pred[-1])
                    left = eval(' '.join(pred[:-2]))
                    if left == right:
                        return 1
        except:
            return 0
        return 0

    def check_token(self, tar, pred):
        min_len = min([len(tar), len(pred)])
        return np.float32(sum(np.equal(tar[:min_len], pred[:min_len]))/len(tar))

    def check_seq(self, tar, pred): 
        min_len = min([len(tar), len(pred)]) 
        if sum(np.equal(tar[:min_len], pred[:min_len])) == len(tar): 
            return 1
        return 0

    def get_eq_acc(self):
        a = 0
        for i in range(self.size):
            tar = self.tars[i]
            pred = self.preds[i]
            a += self.check_equation(tar, pred)
        return np.float32(a/self.size)

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