#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# public
import json
import numpy as np


# helper functions
def load_txt(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f: 
        return f.read().splitlines()

def save_json(path: str, data_dict: dict) -> None:
    with open(path, 'w') as f:
        json.dump(data_dict, f, ensure_ascii=False)

def white_space_tokenizer(str_seq_list: list) -> list:
    return [str_seq.split(' ') for str_seq in str_seq_list]

def convert_to_int(seq:list) -> list:
    return [int(str_number) for str_number in seq]

def convert_to_str(seq:list) -> str:
    return [str(int_number) for int_number in seq]

def find_next_step_in_bubble_sort(seq): 
    n = len(seq) 
    for j in range(0, n-1):
        if seq[j] > seq[j+1]:
            return j
    return -1

def bubble_sort_step(seq, j): 
    # perform one bubble sort step
    seq[j], seq[j+1] = seq[j+1], seq[j] 
    return seq

def gen_recursion_pair(x: str, y: str) -> (list, list):
    # white space tokenization
    x = convert_to_int(x.split())
    y = y.split()
    # record observation
    xs = [x.copy()]
    ys_ = []
    # process bubble sort
    while True:
        y_ = find_next_step_in_bubble_sort(x)
        ys_.append(y_)
        if y_ == -1:
            break
        x = bubble_sort_step(x, y_)
        xs.append(x.copy())

    index = np.random.choice(range(len(xs)))
    return convert_to_str(xs[index]), [str(ys_[index])], y