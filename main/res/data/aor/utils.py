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

def gen_rec_pair(y: list) -> list:
    # white space tokenization
    y = y.split()
    # make a copy
    x = y.copy()
    # get operator indexes
    operator_idxes = [i for i, token in enumerate(y) if not token.isdigit()][::-1]
    # decide how many operators to remove
    num_idxes = np.random.choice(range(len(operator_idxes)+1))
    if num_idxes == 0:
        return x, ['<done>', '<done>'], y
    else:
        # decide operators to remove
        idxes_to_remove = operator_idxes[:num_idxes]
        # generat label
        y_ = ['pos_{}'.format(idxes_to_remove[-1]), x[idxes_to_remove[-1]]]
        # generate sample
        x = [x[i] for i in range(len(x)) if i not in idxes_to_remove]
        
        return x, y_, y

def gen_tag_pair(x, y):
    x_ = x.split()
    y = y.split()
    y_ = []
    x_token = x_.pop(0)
    for i in range(len(y)):    
        y_token = y[i]
        if x_token == y_token:
            y_.append('<keep>')
            if len(x_) == 0:
                break
            x_token = x_.pop(0)
        else:
            y_.append('<insert_{}>'.format(y_token))
    
    return x, ' '.join(y), ' '.join(y_)