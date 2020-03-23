#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'
__email__ = 'ning.shi@learnable.ai'

import os

class Config():
      # config settings
      def __init__(self): 
        # data source
        self.method = "end2end"
        self.model_name = "lstm_rnn"
        self.load_check_point = False
        self.vocab_size = 10
        self.seq_len = 5 # input sequence length
        self.data_size = 10000 # total data size
        # path
        self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
        self.TASK_PATH = os.path.join('vocab_size_{}'.format(self.vocab_size), 
            'seq_len_{}'.format(self.seq_len), 'data_size_{}'.format(self.data_size))
        self.DATA_PATH = os.path.join(self.CURR_PATH, 'res/data/', self.method, self.TASK_PATH, 'data.json')
        self.VOCAB_PATH = os.path.join(self.CURR_PATH, 'res/data/', self.method, self.TASK_PATH, 'vocab.json')
        self.SAVE_PATH = os.path.join(self.CURR_PATH, 'res/check_points/', self.method, self.TASK_PATH)
        if not os.path.exists(self.SAVE_PATH): os.makedirs(self.SAVE_PATH)
        self.SAVE_POINT = os.path.join(self.SAVE_PATH, '{}_step_{}_epoch.pt')
        self.LOAD_POINTS = [p for p in os.listdir(self.SAVE_PATH) if p.endswith('.pt')]
        if len(self.LOAD_POINTS) != 0: 
            self.LOAD_POINT = sorted(self.LOAD_POINTS, key=lambda x: int(x.split('_')[0]))[-1] 
            self.LOAD_POINT = os.path.join(self.SAVE_PATH, self.LOAD_POINT)
        else:
            self.load_check_point = False
        self.LOG_PATH = os.path.join(self.CURR_PATH, 'res/log/', self.method, self.TASK_PATH)
        if not os.path.exists(self.LOG_PATH): os.makedirs(self.LOG_PATH)
        self.LOG_PATH = os.path.join(self.LOG_PATH,  'log.txt')
        # initialization
        self.pad_symbol = '<pad>'
        self.start_symbol = '<s>'
        self.end_symbol = '</s>'
        # data loader
        self.batch_size = 128
        self.shuffle = True
        self.drop_last = True
        # training
        self.learning_rate = 1e-4
        self.teacher_forcing_ratio = 0.5
        self.clipping_threshold = 5
        self.test_epoch = 512
        # model
        # embedding
        self.embedding_size = 512
        # encoder
        self.en_hidden_size = 512
        self.en_num_layers = 1
        # decoder
        self.de_hidden_size = 512
        self.de_num_layers = 1
        # dropout
        self.embedding_drop_rate = 0.5
        self.en_drop_rate = 0.5
        self.de_drop_rate = 0.5

            
