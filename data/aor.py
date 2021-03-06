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


# the calss to generate dataset for Arithmetic Operators Insertion (AOR)
class ArithmeticOperatorRestoration(): 
    """docstring for ArithmeticOperatorRestoration"""
    def __init__(self, operators, N):
        super(ArithmeticOperatorRestoration, self).__init__()
        self.operators = operators
        self.pos_digits_pool = np.arange(2, N+2).tolist()
        self.neg_digits_pool = np.arange(-N, -1).tolist()
        self.digits_pool = self.pos_digits_pool + self.neg_digits_pool
    
    def gen_base_dict(self):
        # initialize a base value dict
        return {str(i):[] for i in self.pos_digits_pool}
        
    def gen_operation(self, L):
        # a recursion to geneate  the left side of an equation
        if L == 1:
            a = np.random.choice(self.digits_pool)
            return [str(a)]
        else:
            left_side  = self.gen_operation(L-1)
            operator = np.random.choice(self.operators)
            b = np.random.choice(self.pos_digits_pool)
            return left_side + [operator, str(b)]
    
    def gen_operation_list(self, L, D):
        # to control the data size
        operations_pool = set()
        for i in tqdm(range(D)):
            while True: 
                # to avoid duplicates
                operation = self.gen_operation(L) 
                if ''.join(operation) in operations_pool: 
                    continue
                else:
                    operations_pool.add(''.join(operation)) 
                # to avoid zero division error
                try: 
                    # flost to int to string
                    value = eval(''.join(operation))
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
                x = x[0].replace('-', '- ').split() + x[1:]
                y = x + ["=="] + [v]
                x = [i for i in y if i.isdigit()]
                self.xs.append(' '.join(x))
                self.ys.append(' '.join(y))

    def generate(self, L, D):
        # input sequences, output sequences
        self.xs, self.ys = [], []
        # initialize a value dictionary
        # to save the value of each left side
        self.value_dict = self.gen_base_dict()
        # generate the left side of an equation
        self.gen_operation_list(
            L=L, 
            D=D)
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
    outdir = 'aor' 
    outdir = os.path.join(
        outdir, 
        '{}N'.format(args.N), 
        '{}L'.format(args.L), 
        '{}D'.format(args.D))
    
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
    # example
    # python aor.py --N 10 --L 5 --D 10000
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', 
        type=int, 
        required=True, 
        help=' defines the number of unique integers')
    parser.add_argument('--L', 
        type=int, 
        required=True, 
        help='defines the number of integers in an equation')
    parser.add_argument('--D', 
        type=int, 
        required=True, 
        help='defines the number of unique equations')
    args = parser.parse_args()
    # data generation 
    operators = ['+', '-', '*', '/']
    moi = ArithmeticOperatorRestoration(operators, args.N) 
    xs, ys = moi.generate(
        L=args.L-1, 
        D=args.D)
    trainset, valset, testset = train_test_split(xs, ys)
    save_dataset(trainset, valset, testset, args)

if __name__ == '__main__': 
    main()