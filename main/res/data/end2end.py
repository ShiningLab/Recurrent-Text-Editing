#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# public
import os
import argparse
import numpy as np
from collections import Counter
# private
from utils import *

class End2EndDataPreprocess(object):
    """docstring for End2EndDataPreprocess"""
    def __init__(self, vocab_size, seq_len, data_size):
        super(End2EndDataPreprocess, self).__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.data_size = data_size
        self.init_paths()
        self.data_preprocess()
        self.train_test_split()
        self.save()

    def init_paths(self):
        # load path
        indir = 'raw'
        self.indir = os.path.join(indir, 'vocab_size_{}'.format(self.vocab_size), 
            'seq_len_{}'.format(self.seq_len), 
            'data_size_{}'.format(self.data_size))
        # save path
        outdir = 'end2end'
        self.outdir = os.path.join(outdir, 'vocab_size_{}'.format(self.vocab_size), 
            'seq_len_{}'.format(self.seq_len), 
            'data_size_{}'.format(self.data_size))
        if not os.path.exists(self.outdir): os.makedirs(self.outdir)

    def data_preprocess(self):
        # load raw dataset
        raw_xs = load_txt(os.path.join(self.indir, 'x.txt'))
        raw_ys = load_txt(os.path.join(self.indir, 'y.txt'))
        # check duplicates
        dataset = [(src, tgt) for src, tgt in zip(raw_xs, raw_ys)]
        dataset = np.array(list(set(dataset)))
        print('dataset shape:', dataset.shape)
        # white space tokenization
        xs = dataset[:, 0]
        ys = dataset[:, 1]
        tk_xs = white_space_tokenizer(xs)
        tk_ys = white_space_tokenizer(ys)
        # vocabulary frequency distribution
        counter = Counter()
        for x in tk_xs: 
            counter.update(x)
        for y in tk_ys:
            counter.update(y)
        vocab_list = sorted(counter.keys())
        # vocabulary dictionary
        self.vocab2idx_dict = dict()
        self.vocab2idx_dict['<pad>'] = 0 # to pad sequence length
        self.vocab2idx_dict['<s>'] = 1 # to mark the start of a sequence
        self.vocab2idx_dict['</s>'] = 2 # to mark the end of a sequence

        i = len(self.vocab2idx_dict)
        for token in vocab_list:
            self.vocab2idx_dict[token] = i
            i += 1
        print('vocab size:', len(self.vocab2idx_dict))
        # convert vocabulary to index
        self.xs = vocab_to_index(tk_xs, self.vocab2idx_dict)
        self.ys = vocab_to_index(tk_ys, self.vocab2idx_dict)

    def train_test_split(self):
        # train test split
        dataset = np.array([(x, y) for x, y in zip(self.xs, self.ys)])
        data_size = dataset.shape[0]
        indices = np.random.permutation(data_size)
        train_size = int(0.8*data_size)
        test_size = int(0.2*data_size)
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        train_set = dataset[train_idx, :]
        test_set = dataset[test_idx, :]
        print('train size', train_size, train_set.shape[0])
        print('test size', test_size, test_set.shape[0])
        # combine data sets to a dict
        train_dict = {}
        train_dict['xs'] = train_set[:, 0].tolist()
        train_dict['ys'] = train_set[:, 1].tolist()

        test_dict = {}
        test_dict['xs'] = test_set[:, 0].tolist()
        test_dict['ys'] = test_set[:, 1].tolist()

        self.data_dict = dict()
        self.data_dict['train'] = train_dict
        self.data_dict['test'] = test_dict

    def save(self):
        # save output as json
        data_path = os.path.join(self.outdir, 'data.json')
        vocab_path = os.path.join(self.outdir, 'vocab.json')
        save_json(data_path, self.data_dict)
        save_json(vocab_path, self.vocab2idx_dict)

def main(): 
    
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', 
        type=int, 
        required=True, 
        help='define the number of positive real digits to involve')
    parser.add_argument('--seq_len', 
        type=int, 
        required=True, 
        help='define the sequence length of inputs')
    parser.add_argument('--data_size', 
        type=int, 
        required=True, 
        help='define the total data size, which can not exceed the space size')
    args = parser.parse_args()
    # data preprocess
    E2EDP = End2EndDataPreprocess(
        vocab_size=args.vocab_size, 
        seq_len=args.seq_len, 
        data_size=args.data_size) 
    
if __name__ == '__main__':
      main()