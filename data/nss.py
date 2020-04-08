#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Ziheng Zeng'


# dependency
# public
import os
import argparse
import numpy as np
from tqdm import tqdm
# private
from utils import *


def generate_end2end(num_size, seq_len, data_size): 
    # generate random number sequence sa sample
    # with its sorted result as label
    xs, ys = [], []
    # to filter out duplicates
    num_seq_set = set()
    for i in tqdm(range(data_size)): 
        while True: 
            # get a random number sequence
            x = np.random.randint(num_size, size=[seq_len])
            # check duplicates
            y = np.sort(x)
            if str(y) in num_seq_set: 
                continue
            else:
                num_seq_set.add(str(y))
                y = np.sort(x)
                # convert a list of int to string
                x = convert_to_str(x)
                y = convert_to_str(y)
                # append to dataset
                xs.append(x)
                ys.append(y)
                break
                    
    return xs, ys

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
    outdir = 'nss' 
    outdir = os.path.join(
        outdir, 
        'num_size_{}'.format(args.num_size), 
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
    parser.add_argument('--num_size', 
        type=int, 
        required=True, 
        help='define the number of real digits to involve')
    parser.add_argument('--seq_len', 
        type=int, 
        required=True, 
        help='define the sequence length of inputs')
    parser.add_argument('--data_size', 
        type=int, 
        required=True, 
        help='define the total data size')
    args = parser.parse_args()
    # data generation 
    xs, ys = generate_end2end(
        num_size=args.num_size, 
        seq_len=args.seq_len, 
        data_size=args.data_size)
    trainset, valset, testset = train_test_split(xs, ys)
    save_dataset(trainset, valset, testset, args)

if __name__ == '__main__': 
    main()