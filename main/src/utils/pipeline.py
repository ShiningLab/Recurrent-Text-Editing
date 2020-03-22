#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


import torch
from torch.utils import data as torch_data


class Een2EndDataset(torch_data.Dataset):
      """Custom data.Dataset compatible with data.DataLoader."""
      def __init__(self, data_dict):
            super(Een2EndDataset, self).__init__()
            self.xs = data_dict['xs']
            self.ys = data_dict['ys']
            self.data_size = len(self.x)

      def __len__(self):
            return self.data_size

      def __getitem__(self, idx):
            return self.xs[idx], self.ys[idx]

def end2end_collate_fn(data, config): 
    # a customized collate function used in the data loader 
    def preprocess(xs, ys): 
        # add start and end symbol 
        xs = [torch.Tensor([self.config.start_idx] + x + [self.config.end_idx]) for x in xs]
        ys_in = [torch.Tensor([self.config.start_idx] + y) for y in ys]
                  ys_out = [torch.Tensor(y + [self.config.end_idx]) for y in ys]
                  return xs, ys_in, ys_out

            def padding(seqs):
                  seq_length_list = [len(seq) for seq in seqs]
                  padded_seqs = torch.zeros([len(seqs), max(seq_length_list)], dtype=torch.int64)
                  for i, seq in enumerate(seqs):
                        seq_length = seq_length_list[i]
                        padded_seqs[i, :seq_length] = seq[:seq_length]
                  return padded_seqs, seq_length_list

            data.sort(key=len, reverse=True)
            xs, ys = zip(*data)
            xs, ys_in, ys_out = preprocess(xs, ys)
            xs, x_lens = padding(xs)
            ys_in, y_lens = padding(ys_in)
            ys_out, _ = padding(ys_out)

            return (xs.to(self.config.device), 
                  torch.Tensor(x_lens).to(self.config.device), 
                  ys_in.to(self.config.device), 
                  ys_out.to(self.config.device), 
                  torch.Tensor(y_lens).to(self.config.device))