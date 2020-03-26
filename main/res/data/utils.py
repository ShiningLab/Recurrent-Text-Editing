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

def vocab_to_index(tk_seq_list: list, vocab_dict: dict) -> list:
    return [[vocab_dict[t] for t in tk_seq] for tk_seq in tk_seq_list]

# a function to generate a sequence pair
# given a label sequence
def get_sequence_pair(y: list) -> list:
    # white space tokenization
    y = y.split()
    x = y.copy()
    # get operator indexes
    operator_idxes = list(range(1, len(x), 2))
    # decide how many operators to remove
    num_idxes = np.random.choice(range(len(operator_idxes)+1))
    if num_idxes == 0:
        return x, ['<completion>', '<none>', '<none>'], y
    else:
        # decide operators to remove
        idxes_to_remove = sorted(np.random.choice(operator_idxes, num_idxes, replace=False))
        # generat possible ys
        ys_ = [['<insertion>', str(idxes_to_remove[i]-i), x[idxes_to_remove[i]]] 
              for i in range(len(idxes_to_remove))]
        # pick y randomly
        y_ = ys_[np.random.choice(range(len(ys_)))]
        # remove operators
        x = [x[i] for i in range(len(x)) if i not in idxes_to_remove]
        return x, y_, y