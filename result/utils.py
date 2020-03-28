#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


import os
import matplotlib.pyplot as plt


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
    items = ['Epoch:', 'Step:', 'Loss:', 'Acc:', 'Token Acc:', 'Seq Acc:']
    for line in in_lines:
        line_dict = dict()
        for i, key in zip(items, keys):
            if i in line:
                i = line.split(i)[1].split()[0]
                line_dict[key] = float(i)
        out_lines.append(line_dict)
    return out_lines

def show_plot(data_dict, colors, title, xlabel, ylabel, save_path): 
    plt.subplots(figsize = (16, 9), dpi=100)
    for key, color in zip(data_dict, colors):
        plt.plot(
            range(len(data_dict[key])), 
            data_dict[key], 
            color=color, 
            linewidth=2, 
            label=key)
        
    plt.title(save_path + ' | ' + title)
    plt.legend()
    plt.xticks(fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    save_path = os.path.join(save_path, title + '.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')