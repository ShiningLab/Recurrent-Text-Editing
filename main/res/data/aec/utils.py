#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# public
import json
import Levenshtein
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

def levenshtein_editops_list(source, target):
    unique_elements = sorted(set(source + target)) 
    char_list = [chr(i) for i in range(len(unique_elements))]
    if len(unique_elements) > len(char_list):
        raise Exception("too many elements")
    else:
        unique_element_map = {ele:char_list[i]  for i, ele in enumerate(unique_elements)}
    source_str = ''.join([unique_element_map[ele] for ele in source])
    target_str = ''.join([unique_element_map[ele] for ele in target])
    transform_list = Levenshtein.editops(source_str, target_str)
    return transform_list

def gen_rec_pair(x: list, y: list) -> list:
    # white space tokenization
    x = x.split()
    y = y.split()
    xs = [x.copy()]
    ys_ = []
    editops = levenshtein_editops_list(x, y)
    if len(editops) == 0: 
        y_ = ['<done>']*3 
    else:
        c = 0 
        for tag, i, j in editops: 
            i += c
            if tag == 'replace':
                y_ = ['<sub>', '<pos_{}>'.format(i), y[j]]
                x[i] = y[j]
            elif tag == 'delete': 
                # y_ = ['<delete>', '<pos_{}>'.format(i), '<done>']
                y_ = ['<delete>', '<pos_{}>'.format(i), '<pos_{}>'.format(i)]
                del x[i]
                c -= 1
            elif tag == 'insert': 
                y_ = ['<insert>', '<pos_{}>'.format(i), y[j]]
                x.insert(i, y[j]) 
                c += 1
            xs.append(x.copy()) 
            ys_.append(y_)
        # ys_.append(['<done>']*3)
        index = np.random.choice(range(len(xs)-1))
        x = xs[index]
        y_ = ys_[index]
    return x, y_, y

def gen_tag_pair(x, y):
    x = x.split()
    y = y.split()
    editops = levenshtein_editops_list(x, y)
    y_ = ['<keep>'] * len(x)
    c = 0
    for tag, i, j in editops:
        i += c
        if tag == 'replace': 
            if y_[i] != '<keep>':
                y_.insert(i+1, '<sub_{}>'.format(y[j]))
                c += 1
            else:
                y_[i] = '<sub_{}>'.format(y[j])
        elif tag == 'delete':
            y_[i] = '<delete>'
        elif tag == 'insert': 
            y_.insert(i, '<insert_{}>'.format(y[j]))
            c += 1
    return x, y, y_