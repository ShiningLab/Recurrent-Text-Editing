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

def gen_recursion_pair(x: str, y: str) -> (list, list):
    # white space tokenization
    x = convert_to_int(x.split())
    y = convert_to_int(y.split())
    # record observation
    xs = [x.copy()]
    ys_ = []
    # process swap sort
    while True:
        min_idx = np.argmin(x)
        ys_.append([min_idx])
        del x[min_idx]
        if len(x) == 0:
            break
        xs.append(x.copy())    
    index = np.random.choice(range(len(xs)))

    return convert_to_str(xs[index]), convert_to_str(ys_[index]), convert_to_str(y)

def nss_generator(ys: list) -> list:
    xs = []
    for y in ys:
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        x = np.array(y)[idx].tolist()
        xs.append(x)
    return [(x, y) for x, y in zip(xs, ys)]