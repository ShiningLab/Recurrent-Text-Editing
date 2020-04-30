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
            y_.append('<sub_{}>'.format(y_token))
            x_token = x_.pop(0)
            while True:
                y_.append('<delete>')
                if x_token == ')':
                    if len(x_) != 0:
                        x_token = x_.pop(0)
                    break
                x_token = x_.pop(0)
    return x, ' '.join(y), ' '.join(y_)