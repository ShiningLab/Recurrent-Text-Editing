#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'ning.shi@learnable.ai'

import os

class Config():
      # config settings
      def __init__(self):
             # setting
            self.reuse_model = False
            self.general_show = True
            # model
            # seq2seq_bi_lstm_att
            self.model_name = 'seq2seq_bi_lstm_att'
            # data set
            # gen/non_ch, gen/ch, gen/mixed, real
            self.data_source = 'gen/mixed'
            # path
            self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
            self.DATA_PATH = os.path.join(self.CURR_PATH, 'res/data/{}/data.json'.format(self.data_source))
            self.VOCAB_PATH = os.path.join(self.CURR_PATH, 'res/data/vocab.json')
            self.SAVE_PATH = os.path.join(self.CURR_PATH, 'res/log/{}/{}/'.format(self.data_source, self.model_name))
            if not os.path.exists(self.SAVE_PATH): os.makedirs(self.SAVE_PATH)
            # step, epoch
            self.SAVE_POINT = os.path.join(self.SAVE_PATH, '{}_step_{}_epoch.pt')
            self.LOAD_POINTS = [p for p in os.listdir(self.SAVE_PATH) if p.endswith('.pt')]
            if len(self.LOAD_POINTS) != 0:
                  self.LOAD_POINT = sorted(self.LOAD_POINTS, key=lambda x: int(x.split('_')[0]))[-1]
                  self.LOAD_POINT = os.path.join(self.SAVE_PATH, self.LOAD_POINT)
            else:
                  self.reuse_model = False
            # data preprocess
            self.shuffle = True
            self.drop_last = True 
            # train
            self.early_stop = 'seq_acc' # loss, token_acc, seq_acc
            self.data_parallel = False # enable data parallel training
            self.batch_size = 128
            self.teacher_forcing_ratio = 0.5
            self.progress_rate = 1/4
            self.test_epoch = 512
            # for data generation API
            self.train_data_size = self.test_epoch * self.batch_size
            self.sub_data_size = int(self.train_data_size/3)
            # initialization
            self.pad_symbol = '<pad>'
            self.start_symbol = '<s>'
            self.end_symbol = '</s>'
            # embedding layer
            self.embedding_size = 512
            # encoding layer
            self.en_num_units = 256
            self.en_num_layers = 2
            # decoding layer
            self.de_num_units = 512
            self.de_num_layers = 1
            # dropout layer
            self.embedding_drop_rate = 0.5
            self.en_drop_rate = 0.5
            self.de_drop_rate = 0.5
            # optimizer
            self.clipping_threshold = 5
            self.learning_rate = 1e-4

class InferConfig(Config):
      """docstring for InferConfig"""
      def __init__(self):
            super(InferConfig, self).__init__()
            # settings for inference
            self.input_file = "input.txt"
            self.output_file = "output.txt"
            self.model_name = 'seq2seq_bi_lstm_att'
            self.data_source = 'gen/mixed'
            self.checkpoint_file = "167475_step_384_epoch.pt"
            self.max_tar_len = 100

            
