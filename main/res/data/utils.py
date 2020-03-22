#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# helper functions
def load_txt(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f: 
        return f.read().splitlines()

def white_space_tokenizer(str_seq_list: list) -> list:
    return [str_seq.split(' ') for str_seq in str_seq_list]

def vocab_to_index(tk_seq_list: list, vocab_dict: dict) -> list:
    return [[vocab_dict[t] for t in tk_seq] for tk_seq in tk_seq_list]