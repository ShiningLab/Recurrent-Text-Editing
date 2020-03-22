#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
import os
import argparse
import numpy as np
from utilities.save import save_txt


# the calss to generate dataset
# for math operator intertion task
class MathematicalOperatorInsertion(): 
    """docstring for ClassName"""
    def __init__(self):
        super(MathematicalOperatorInsertion, self).__init__()
        self.operators = ['+', '-', '*', '/']
    
    def gen_base_dataset(self, vocab_size):
        # return a base dataset
        x = [str(i) for i in range(vocab_size)]
        y = x.copy()
        return x, y
    
    def gen_base_dict(self, vocab_size):
        # initialize a base value dict
        return {str(i):[] for i in range(vocab_size)}
        
    def gen_operation(self, vocab_size, seq_len):
        # a recursive function to geneate an operation
        # given the number of digits to involve
        a = np.random.choice(range(vocab_size))
        o = np.random.choice(self.operators)
        b = np.random.choice(range(vocab_size))
        if seq_len == 1:
            return [str(a)]
        else:
            out_set = self.gen_operation(vocab_size, seq_len-1)
            return out_set + [o, str(b)]
    
    def gen_operation_list(self, vocab_size, seq_len, data_size):
        # to control the data size
        counter = 1
        operations_pool = set()
        while True:
            # to avoid duplicates
            operation = self.gen_operation(vocab_size, seq_len)
            if ''.join(operation) in operations_pool:
                continue
            else:
                operations_pool.add(''.join(operation))
            # to avoid zero division error
            try: 
                value = str(eval(' '.join(operation))) 
                # to keep vocab size
                if value in self.value_dict: 
                    self.value_dict[value].append(operation)
                    if counter >= data_size:
                        break
                    else:
                        counter += 1
            except: 
                pass
            if len(operations_pool) >= self.space_size: 
                break
    
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
        # the max data size
        self.space_size = vocab_size**seq_len*len(self.operators)**(seq_len-1)
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

def save_dataset(xs, ys, args):

    data_size = len(xs)
    print("real data size", len(ys), data_size)

    outdir = 'raw'
    outdir = os.path.join(outdir, 'vocab_size_{}'.format(args.vocab_size), 
                    'seq_len_{}'.format(args.seq_len), 
                    'data_size_{}'.format(data_size))
    
    if not os.path.exists(outdir): 
        os.makedirs(outdir)

    save_txt(os.path.join(outdir, 'x.txt'), xs)
    save_txt(os.path.join(outdir, 'y.txt'), ys)

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
    moi = MathematicalOperatorInsertion() 
    xs, ys = moi.generate(
        vocab_size=args.vocab_size, 
        seq_len=args.seq_len-1, 
        data_size=args.data_size)

    save_dataset(xs, ys, args)

if __name__ == '__main__': 
    main()