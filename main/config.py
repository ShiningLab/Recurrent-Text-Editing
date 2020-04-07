#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'

import os

class Config():
      # config settings
      def __init__(self): 
        # data source
        self.data_src = 'nss' # aoi, nss
        self.method = 'end2end' # end2end, recursion, tagging
        self.data_mode = 'online' # online, offline 
        # gru_rnn, lstm_rnn, bi_gru_rnn, bi_lstm_rnn, 
        # bi_gru_rnn_att, bi_lstm_rnn_att
        # gru_ptr, lstm_ptr, bi_gru_ptr, bi_lstm_ptr
        self.model_name = 'gru_ptr'
        self.load_check_point = False
        self.num_size = 100 # numbers involved
        self.seq_len = 5 # input sequence length
        self.data_size = 10000 # total data size
        # I/O directory
        # current path
        self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
        # data task path
        self.TASK_PATH = os.path.join('num_size_{}'.format(self.num_size), 
            'seq_len_{}'.format(self.seq_len), 'data_size_{}'.format(self.data_size))
        # data dictionary in json file
        self.DATA_PATH = os.path.join(self.CURR_PATH, 'res/data/', 
            self.data_src, self.method, self.TASK_PATH, 'data.json')
        # vocab dictionary in json file
        self.VOCAB_PATH = os.path.join(self.CURR_PATH, 'res/data/', 
            self.data_src, self.method, self.TASK_PATH, 'vocab.json')
        # path to save and load check point
        self.SAVE_PATH = os.path.join(self.CURR_PATH, 'res/check_points/', 
            self.data_src, self.data_mode, self.method, self.TASK_PATH)
        if not os.path.exists(self.SAVE_PATH): os.makedirs(self.SAVE_PATH)
        self.SAVE_POINT = os.path.join(self.SAVE_PATH, '{}.pt'.format(self.model_name))
        self.LOAD_POINT = self.SAVE_POINT
        if not os.path.exists(self.LOAD_POINT): self.load_check_point = False
        # path to save test log
        self.LOG_PATH = os.path.join(self.CURR_PATH, 'res/log/', 
            self.data_src, self.data_mode, self.model_name, self.method, self.TASK_PATH)
        if not os.path.exists(self.LOG_PATH): os.makedirs(self.LOG_PATH)
        self.LOG_POINT = os.path.join(self.LOG_PATH,  '{}.txt')
        # path to save test output
        self.RESULT_PATH = os.path.join(self.CURR_PATH, 'res/result/', 
            self.data_src, self.data_mode, self.model_name, self.method, self.TASK_PATH)
        if not os.path.exists(self.RESULT_PATH): os.makedirs(self.RESULT_PATH)
        self.RESULT_POINT = os.path.join(self.RESULT_PATH, '{}.txt')
        # initialization
        self.pad_symbol = '<pad>'
        self.start_symbol = '<s>'
        self.end_symbol = '</s>'
        # data loader
        self.batch_size = 256
        self.shuffle = True
        self.drop_last = True
        # val
        self.val_win_size = 512
        # model
        self.learning_rate = 1e-4
        self.teacher_forcing_ratio = 0.5
        self.clipping_threshold = 5
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

class End2EndConfig(Config):
    """docstring for End2EndConfig"""
    def __init__(self):
        super(End2EndConfig, self).__init__()
    
class RecursionConfig(Config):
    """docstring for RecursionConfig"""
    def __init__(self):
        super(RecursionConfig, self).__init__()
        # define the max inference step
        if self.data_src == 'aoi':
            self.max_infer_step = self.seq_len
            self.tgt_seq_len = 3 # action, position, target operator
        elif self.data_src == 'nss':
            self.max_infer_step = self.seq_len
            self.tgt_seq_len = 2 # src_idx_to_swap, tgt_idx_to_swap

class TaggingConfig(Config):
    """docstring for TaggingConfig"""
    def __init__(self):
        super(TaggingConfig, self).__init__()