#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# public
import os
import argparse
import numpy as np
from tqdm import tqdm
# private
from utils import *


# the calss to generate dataset
# for math operator intertion task
class MathematicalOperatorInsertion(): 
    """docstring for ClassName"""
    def __init__(self, operators):
        super(MathematicalOperatorInsertion, self).__init__()
        self.operators = operators
    
    def gen_base_dataset(self, vocab_size):
        # return a base dataset
        x = [str(i) for i in range(2, vocab_size+2)]
        y = x.copy()
        return x, y
    
    def gen_base_dict(self, vocab_size):
        # initialize a base value dict
        return {str(i):[] for i in range(2, vocab_size+2)}
        
    def gen_operation(self, vocab_size, seq_len):
        # a recursive function to geneate an operation
        # given the number of digits to involve
        a = np.random.choice(range(2, vocab_size+2))
        if seq_len == 1:
            return [str(a)]
        else:
            out_set = self.gen_operation(vocab_size, seq_len-1)
            o = np.random.choice(self.operators)
            b = np.random.choice(range(2, vocab_size+2))
            return out_set + [o, str(b)]
    
    def gen_operation_list(self, vocab_size, seq_len, data_size):
        # to control the data size
        operations_pool = set()
        for i in tqdm(range(data_size)):
            while True: 
                # to avoid duplicates
                operation = self.gen_operation(vocab_size, seq_len) 
                if ''.join(operation) in operations_pool: 
                    continue
                else:
                    operations_pool.add(''.join(operation)) 
                # to avoid zero division error
                try: 
                    # flost to int to string
                    value = eval(' '.join(operation)) 
                    if value % 1 != 0.: 
                        continue
                    else:
                        value = str(int(value))
                        # to keep vocab size
                        if value in self.value_dict: 
                            self.value_dict[value].append(operation)
                            break
                except: 
                    pass
    
    def gen_equation_list(self):
        # generate the relational equation
        # given the value dict
        for v in self.value_dict:
            for x in self.value_dict[v]:
                y = x + ["=="] + [v]
                x = [i for i in y if i.isdigit()]
                self.xs.append(' '.join(x))
                self.ys.append(' '.join(y))

    def generate(self, vocab_size, seq_len, data_size):
        if seq_len == 0:
            return self.gen_base_dataset(
                vocab_size=vocab_size)
        # input sequences, # output sequences
        self.xs, self.ys = [], []
        # initialize a value dictionary
        # to save the value of each sequence
        self.value_dict = self.gen_base_dict(
            vocab_size=vocab_size)
        # insert operators and generate equations
        self.gen_operation_list(
            vocab_size=vocab_size, 
            seq_len=seq_len, 
            data_size=data_size)
        # generate relations given the value dict
        self.gen_equation_list()
        
        return self.xs, self.ys

def train_test_split(xs, ys): 
    # train val test split
    dataset = np.array([(x, y) for x, y in zip(xs, ys)])
    data_size = dataset.shape[0]
    indices = np.random.permutation(data_size)
    train_size = int(0.7*data_size)
    val_size = int(0.15*data_size)
    test_size = data_size - train_size - val_size
    train_idxes = indices[:train_size]
    val_idxes = indices[train_size: train_size+val_size]
    test_idxes = indices[train_size+val_size:]
    trainset = dataset[train_idxes]
    valset = dataset[val_idxes]
    testset = dataset[test_idxes]
    print('train size', train_size, trainset.shape)
    print('val size', val_size, valset.shape)
    print('test size', test_size, testset.shape)

    return trainset, valset, testset

def save_dataset(trainset, valset, testset, args): 
    outdir = 'raw' 
    outdir = os.path.join(
        outdir, 
        'vocab_size_{}'.format(args.vocab_size), 
        'seq_len_{}'.format(args.seq_len), 
        'data_size_{}'.format(args.data_size))
    
    if not os.path.exists(outdir): 
        os.makedirs(outdir)

    save_txt(os.path.join(outdir, 'train_x.txt'), trainset[:, 0])
    save_txt(os.path.join(outdir, 'train_y.txt'), trainset[:, 1])
    save_txt(os.path.join(outdir, 'val_x.txt'), valset[:, 0])
    save_txt(os.path.join(outdir, 'val_y.txt'), valset[:, 1])
    save_txt(os.path.join(outdir, 'test_x.txt'), testset[:, 0])
    save_txt(os.path.join(outdir, 'test_y.txt'), testset[:, 1])

    print("find output from", outdir)

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
    # data generation 
    operators = ['+', '-', '*', '/'] #TODO
    moi = MathematicalOperatorInsertion(operators) 
    xs, ys = moi.generate(
        vocab_size=args.vocab_size, 
        seq_len=args.seq_len-1, 
        data_size=args.data_size)
    trainset, valset, testset = train_test_split(xs, ys)
    save_dataset(trainset, valset, testset, args)

if __name__ == '__main__': 
    main()