#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


#public
import torch
from torch.utils import data as torch_data

import random
# private
from ..models import gru_rnn


class Een2EndDataset(torch_data.Dataset):
      """Custom data.Dataset compatible with data.DataLoader."""
      def __init__(self, data_dict):
            super(Een2EndDataset, self).__init__()
            self.xs = data_dict['xs']
            self.ys = data_dict['ys']
            self.data_size = len(self.xs)

      def __len__(self):
            return self.data_size

      def __getitem__(self, idx):
            return self.xs[idx], self.ys[idx]

def pick_model(model_name, config):
    if model_name == "gru_rnn":
        return gru_rnn.ModelGraph(config).to(config.device)

def init_parameters(model): 
    for name, parameters in model.named_parameters(): 
        if 'weight' in name: 
            torch.nn.init.normal_(parameters.data, mean=0, std=0.01)
        else:
            torch.nn.init.constant_(parameters.data, 0)

def count_parameters(model): 
    # get total size of trainable parameters 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def show_config(config, model):
    # show configuration
    print('\n*Configuration*')
    print('model:', config.model_name)
    print('trainable parameters:{:,.0f}'.format(config.num_parameters))
    print("model's state_dict:")
    for parameters in model.state_dict(): 
        print(parameters, "\t", model.state_dict()[parameters].size())
    print('device:', config.device)
    print('use gpu:', config.use_gpu)
    print('train size:', config.train_size)
    print('test size:', config.test_size)
    print('vocab size:', config.vocab_size)
    print('batch size:', config.batch_size)
    print('train batch:', config.train_batch)
    print('test batch:', config.test_batch)
    print('\nif load check point:', config.load_check_point)
    if config.load_check_point: 
        print('Model restored from {}.'.format(config.LOAD_POINT))
        print()

def index_to_vocab(idx_seq: list, idx2vocab_dict: dict) -> list: 
    return [idx2vocab_dict[idx] for idx in idx_seq]

def vocab_to_index(tk_seq: list, vocab2idx_dict:dict) -> list:
    return [vocab2idx_dict[token] for token in tk_seq]

def rm_pad(seq, pad_idx):
    return [i for i in seq if i != pad_idx]

def prepare_output(srcs, tars, preds, pad_idx): 
    srcs = [rm_pad(seq, pad_idx) for seq in srcs] 
    tars = [rm_pad(seq, pad_idx) for seq in tars] 
    preds = [p_seq[:len(t_seq)] for p_seq, t_seq in zip(preds, tars)] 
    return srcs, tars, preds

def save_check_point(step, epoch, model_state_dict, opt_state_dict, path):
    # save model, optimizer, and everything required to keep
    checkpoint_to_save = {
        'step': step, 
        'epoch': epoch, 
        'model': model_state_dict(), 
        'optimizer': opt_state_dict()}
    torch.save(checkpoint_to_save, path)
    print('Model saved as {}.'.format(path))

def rand_sample(srcs, tars, preds, idx2vocab_dict): 
    src, tar, pred = random.choice([(src, tar, pred) for src, tar, pred in zip(srcs, tars, preds)])
    src = index_to_vocab(src, idx2vocab_dict)
    tar = index_to_vocab(tar, idx2vocab_dict)
    pred = index_to_vocab(pred, idx2vocab_dict)
    return ' '.join(src), ' '.join(tar), ' '.join(pred)

