#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# helper functions
def save_txt(path: str, line_list:list) -> None:
    with open(path, 'w', encoding='utf-8') as f: 
        for line in line_list: 
            f.write(line + '\n') 
    f.close()

def load_txt(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f: 
        return f.read().splitlines()

def parse_log(in_lines):
    out_lines = []
    keys = ['epoch', 'step', 'loss', 'acc', 'token_acc', 'seq_acc']
    for line in in_lines:
        line_dict = dict()
        for i, key in zip(line.split(':')[1:], keys):
            line_dict[key] = float(i.split()[0])
        out_lines.append(line_dict)
    return out_lines