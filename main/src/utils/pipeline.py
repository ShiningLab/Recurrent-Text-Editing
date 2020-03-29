#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


#public
import torch
from torch.utils import data as torch_data

import random
import numpy as np

# private
from ..models import gru_rnn, lstm_rnn, bi_lstm_rnn_att


class Dataset(torch_data.Dataset):
      """Custom data.Dataset compatible with data.DataLoader."""
      def __init__(self, data_dict):
            super(Dataset, self).__init__()
            self.xs = data_dict['xs']
            self.ys = data_dict['ys']
            self.data_size = len(self.xs)

      def __len__(self):
            return self.data_size

      def __getitem__(self, idx):
            return self.xs[idx], self.ys[idx]

# class Een2EndDataset(torch_data.Dataset):
#       """Custom data.Dataset compatible with data.DataLoader."""
#       def __init__(self, data_dict):
#             super(Een2EndDataset, self).__init__()
#             self.xs = data_dict['xs']
#             self.ys = data_dict['ys']
#             self.data_size = len(self.xs)

#       def __len__(self):
#             return self.data_size

#       def __getitem__(self, idx):
#             return self.xs[idx], self.ys[idx]

# class OfflineRecursionDataset(torch_data.Dataset):
#       """Custom data.Dataset compatible with data.DataLoader."""
#       def __init__(self, data_dict):
#             super(OfflineRecursionDataset, self).__init__()
#             self.xs = data_dict['xs']
#             self.ys = data_dict['ys']
#             self.data_size = len(self.xs)

#       def __len__(self):
#             return self.data_size

#       def __getitem__(self, idx):
#             return self.xs[idx], self.ys[idx]

class OnlineRecursionDataset(torch_data.Dataset):
      """Custom data.Dataset compatible with data.DataLoader."""
      def __init__(self, data_dict):
            super(OnlineRecursionDataset, self).__init__()
            self.ys = data_dict['ys']
            self.data_size = len(self.ys)

      def __len__(self):
            return self.data_size

      def __getitem__(self, idx):
            return self.ys[idx]

def pick_model(config, method):
    if config.model_name == 'gru_rnn':
        if method == 'end2end':
            return gru_rnn.End2EndModelGraph(config).to(config.device)
        elif method == 'recursion':
            return gru_rnn.RecursionModelGraph(config).to(config.device)
    elif config.model_name == 'lstm_rnn':
        if method == 'end2end':
            return lstm_rnn.End2EndModelGraph(config).to(config.device)
        elif method == 'recursion':
            return lstm_rnn.RecursionModelGraph(config).to(config.device)
    elif config.model_name =='bi_lstm_rnn_att':
        if method == 'end2end':
            return bi_lstm_rnn_att.End2EndModelGraph(config).to(config.device)
        if method == 'recursion':
            return bi_lstm_rnn_att.RecursionModelGraph(config).to(config.device)

def get_list_mean(l: list) -> float:
    return sum(l) / len(l)

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
    print('val size:', config.val_size)
    print('test size:', config.test_size)
    print('source vocab size:', config.src_vocab_size)
    print('target vocab size:', config.tgt_vocab_size)
    print('batch size:', config.batch_size)
    print('train batch:', config.train_batch)
    print('val batch:', config.val_batch)
    print('test batch:', config.test_batch)
    print('\nif load check point:', config.load_check_point)
    if config.load_check_point: 
        print('Model restored from {}.'.format(config.LOAD_POINT))
        print()

def translate(idx_seq: list, idx2vocab_dict: dict) -> list: 
    return [idx2vocab_dict[idx] for idx in idx_seq]

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

def rand_sample(srcs, tars, preds, src_dict, tar_dict, pred_dict): 
    src, tar, pred = random.choice([(src, tar, pred) for src, tar, pred in zip(srcs, tars, preds)])
    src = translate(src, src_dict)
    tar = translate(tar, tar_dict)
    pred = translate(pred, pred_dict)
    return ' '.join(src), ' '.join(tar), ' '.join(pred)

# # a function to generate a sequence pair
# # given a label sequence
# def get_sequence_pair(y: list) -> list:
#     x = y.copy()
#     # get operator indexes
#     operator_idxes = list(range(1, len(x), 2))
#     # decide how many operators to remove
#     num_idxes = np.random.choice(range(len(operator_idxes)+1))
#     if num_idxes == 0:
#         return x, ['<completion>', '<none>', '<none>']
#     else:
#         # decide operators to remove
#         idxes_to_remove = sorted(np.random.choice(operator_idxes, num_idxes, replace=False))
#         # generat possible ys
#         ys = [['<insertion>', str(idxes_to_remove[i]-i), x[idxes_to_remove[i]]] 
#               for i in range(len(idxes_to_remove))]
#         # pick y randomly
#         y = ys[np.random.choice(range(len(ys)))]
#         # remove operators
#         x = [x[i] for i in range(len(x)) if i not in idxes_to_remove]
#         return x, y

def get_sequence_pair(y: list) -> list:
    # white space tokenization
    x = y.copy()
    # get operator indexes
    operator_idxes = np.arange(1, len(x), 2)[::-1]
    # decide how many operators to remove
    num_idxes = np.random.choice(range(len(operator_idxes)+1))
    if num_idxes == 0:
        return x, ['<completion>', '<none>', '<none>']
    else:
        # decide operators to remove
        idxes_to_remove = operator_idxes[:num_idxes]
        y = ['<insertion>', str(idxes_to_remove[-1]), x[idxes_to_remove[-1]]]
        x = [x[i] for i in range(len(x)) if i not in idxes_to_remove]
        return x, y

def preprocess(xs, ys, src_vocab2idx_dict, tgt_vocab2idx_dict, end_idx): 
    # vocab to index
    xs = [translate(x, src_vocab2idx_dict) for x in xs]
    ys = [translate(y, tgt_vocab2idx_dict) for y in ys]
    # add start and end symbol 
    xs = [torch.Tensor(x + [end_idx]) for x in xs]
    ys = [torch.Tensor(y + [end_idx]) for y in ys] 
    return xs, ys

def padding(seqs, max_len=None):
    # zero padding
    seq_lens = [len(seq) for seq in seqs]
    if max_len is None:
        max_len = max(seq_lens)
    # default pad index is 0
    padded_seqs = torch.zeros([len(seqs), max_len], dtype=torch.int64)
    for i, seq in enumerate(seqs): 
        seq_len = seq_lens[i]
        padded_seqs[i, :seq_len] = seq[:seq_len]
    return padded_seqs, seq_lens

def recursive_infer(xs, x_lens, ys_, src_idx2vocab_dict, src_vocab2idx_dict, tgt_idx2vocab_dict, config):
    # detach from devices
    xs = xs.cpu().detach().numpy() 
    ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy()
    # remove padding
    xs = [rm_pad(x, config.pad_idx) for x in xs] 
    # convert index to vocab
    xs = [translate(x, src_idx2vocab_dict) for x in xs]
    ys_ = [translate(y_, tgt_idx2vocab_dict) for y_ in ys_]
    for x, y_ in zip(xs, ys_): 
        if y_[0] == '<insertion>' and y_[1].isdigit() and y_[2] in ['+', '-', '*', '/', '==']: 
            x.insert(int(y_[1]), y_[2])
    xs = [torch.Tensor(translate(x, src_vocab2idx_dict)) for x in xs]
    xs, x_lens = padding(xs, config.seq_len*2)
    return xs.to(config.device), torch.Tensor(x_lens).to(config.device), 
