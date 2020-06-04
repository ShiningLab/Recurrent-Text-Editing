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


class TaggingDataPreprocess(object):
    """docstring for TaggingDataPreprocess"""
    def __init__(self, N, L, D):
        super(TaggingDataPreprocess, self).__init__() 
        self.method = 'tag'
        self.N = N
        self.L = L
        self.D = D
        self.init_paths()
        self.data_preprocess()
        self.save()

    def init_paths(self):
        # load path
        indir = 'aec'
        self.indir = os.path.join(
            indir, 
            '{}N'.format(self.N), 
            '{}L'.format(self.L), 
            '{}D'.format(self.D))
        # save path
        self.outdir = os.path.join(self.method, 
            '{}N'.format(self.N), 
            '{}L'.format(self.L), 
            '{}D'.format(self.D))
        if not os.path.exists(self.outdir): os.makedirs(self.outdir)

    def data_preprocess(self):
        # load raw dataset
        raw_train_xs = load_txt(os.path.join(self.indir, 'train_x.txt'))
        raw_train_ys = load_txt(os.path.join(self.indir, 'train_y.txt'))
        raw_val_xs = load_txt(os.path.join(self.indir, 'val_x.txt'))
        raw_val_ys = load_txt(os.path.join(self.indir, 'val_y.txt'))
        raw_test_xs = load_txt(os.path.join(self.indir, 'test_x.txt'))
        raw_test_ys = load_txt(os.path.join(self.indir, 'test_y.txt'))
        # check data size
        print('train sample size', len(raw_train_xs))
        print('train label size', len(raw_train_ys))
        print('val sample size', len(raw_val_xs))
        print('val label size', len(raw_val_ys))
        print('test sample size', len(raw_test_xs))
        print('test label size', len(raw_test_ys))
        # train
        train_xs, train_ys, train_ys_ = zip(*[gen_tag_pair(x, y) for x, y in zip(raw_train_xs, raw_train_ys)])
        # source vocabulary dictionary
        counter = Counter()
        for x in train_xs:
            counter.update(x)
        src_vocab_list = sorted(counter.keys())
        # soruce vocabulary dictionary
        src_vocab2idx_dict = dict()
        src_vocab2idx_dict['<pad>'] = 0 # to pad sequence length
        i = len(src_vocab2idx_dict)
        for token in src_vocab_list:
            src_vocab2idx_dict[token] = i
            i += 1
        # target vocabulary dictionary
        counter = Counter()
        for y_ in train_ys_:
            counter.update(y_)
        tgt_vocab_list = sorted(counter.keys())
        tgt_vocab2idx_dict = dict()
        tgt_vocab2idx_dict['<pad>'] = 0 # to pad sequence length
        tgt_vocab2idx_dict['<s>'] = 1 # to mark the start of a sequence
        tgt_vocab2idx_dict['</s>'] = 2 # to mark the end of a sequence
        i = len(tgt_vocab2idx_dict)
        for token in tgt_vocab_list:
            tgt_vocab2idx_dict[token] = i
            i += 1
        # val
        # white space tokenization
        val_xs = white_space_tokenizer(raw_val_xs)
        val_ys = white_space_tokenizer(raw_val_ys)
        # test
        # white space tokenization
        test_xs = white_space_tokenizer(raw_test_xs)
        test_ys = white_space_tokenizer(raw_test_ys)
        # combine data sets to a dict
        train_dict = {}
        train_dict['ys'] = train_ys

        val_dict = {}
        val_dict['xs'] = val_xs
        val_dict['ys'] = val_ys

        test_dict = {}
        test_dict['xs'] = test_xs
        test_dict['ys'] = test_ys

        data_dict = dict()
        data_dict['train'] = train_dict
        data_dict['val'] = val_dict
        data_dict['test'] = test_dict

        vocab_dict = dict()
        vocab_dict['src'] = src_vocab2idx_dict
        vocab_dict['tgt'] = tgt_vocab2idx_dict

        self.data_dict = dict()
        self.data_dict['train'] = train_dict
        self.data_dict['val'] = val_dict
        self.data_dict['test'] = test_dict

        self.vocab_dict = dict()
        self.vocab_dict['src'] = src_vocab2idx_dict
        self.vocab_dict['tgt'] = tgt_vocab2idx_dict

    def save(self):
        # save output as json
        data_path = os.path.join(self.outdir, 'data.json')
        vocab_path = os.path.join(self.outdir, 'vocab.json')

        save_json(data_path, self.data_dict)
        save_json(vocab_path, self.vocab_dict)

def main(): 
    # example
    # python tag.py --N 10 --L 5 --D 10000
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', 
        type=int, 
        required=True, 
        help='defines the number of unique integers')
    parser.add_argument('--L', 
        type=int, 
        required=True, 
        help='defines the number of integers in an equation')
    parser.add_argument('--D', 
        type=int, 
        required=True, 
        help='defines the number of unique equations')
    args = parser.parse_args()
    # data preprocess
    TDP = TaggingDataPreprocess(
        N=args.N, 
        L=args.L, 
        D=args.D) 
    
if __name__ == '__main__':
      main()