#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


import torch
import random
import numpy as np


class ModelGraph(torch.nn.Module):
      """docstring for ModelGraph"""
      def __init__(self, config):
            super(ModelGraph, self).__init__()
            self.config = config
            self.encoder = Encoder(config)
            self.decoder = AttDecoder(config)

      def forward(self, x, x_len, y_in, y_out, y_len, teacher_forcing_ratio=0.5, train=True):
            # x: batch size, max src len
            # y_in: batch size, max tar len
            batch_size, max_src_len= x.size()
            max_tar_len = y_out.size()[1] if train else self.config.max_tar_len
            en_output, de_hidden = self.encoder(x)
            # batch size
            tar_token = y_in[:, 0]
            # greedy search
            outputs = torch.zeros(max_tar_len, batch_size, self.config.vocab_size).to(self.config.device)
            for t in range(max_tar_len):
                  de_output, de_hidden = self.decoder(tar_token, de_hidden, en_output, max_src_len, x_len)
                  outputs[t] = de_output
                  tar_token = y_out[:, t] if random.random() < teacher_forcing_ratio else de_output.max(1)[1]
            # batch size, max tar len, tar vocab size
            return outputs.transpose(0, 1).contiguous()

class Encoder(torch.nn.Module):
      """docstring for Encoder"""
      def __init__(self, config):
            super(Encoder, self).__init__()
            self.config = config
            self.embedding = torch.nn.Embedding(
                  num_embeddings=self.config.vocab_size, 
                  embedding_dim=self.config.embedding_size, 
                  padding_idx=self.config.pad_idx)
            self.em_dropout = torch.nn.Dropout(self.config.embedding_drop_rate)
            self.lstm = torch.nn.LSTM(
                  input_size=self.config.embedding_size, 
                  hidden_size=self.config.en_num_units, 
                  num_layers=self.config.en_num_layers, 
                  batch_first=True, 
                  dropout=self.config.en_drop_rate, 
                  bidirectional=True)
            self.en_dropout = torch.nn.Dropout(self.config.en_drop_rate)
            self.h_out = torch.nn.Linear(self.config.en_num_units*self.config.en_num_layers*2, self.config.de_num_units)
            self.c_out = torch.nn.Linear(self.config.en_num_units*self.config.en_num_layers*2, self.config.de_num_units)
      
      def forward(self, input):
            # input: batch_size, max_en_seq_len
            # batch size, seq len, embed dim
            embedded = self.embedding(input)
            embedded = self.em_dropout(embedded)
            # batch_size, max_seq_len, en_hidden_size*num_directions
            # num_layers*num_directions, batch_size, en_hidden_size
            self.lstm.flatten_parameters()
            en_output, (h, c) = self.lstm(embedded)
            en_output = self.en_dropout(en_output)
            # 1, batch, en_hidden_size*num_layers*num_directions
            h = torch.unsqueeze(torch.cat(torch.unbind(h, 0), 1), 0)
            c = torch.unsqueeze(torch.cat(torch.unbind(c, 0), 1), 0)
            # 1, batch, de_hidden_size
            return en_output, (self.h_out(h), self.c_out(h))

class Attn(torch.nn.Module):
      def __init__(self, config):
            super(Attn, self).__init__()
            self.config = config
            self.attn = torch.nn.Linear(self.config.de_num_units*3, self.config.de_num_units)
            self.v = torch.nn.Parameter(torch.rand(self.config.de_num_units))
            self.v.data.normal_(mean=0, std=1./np.sqrt(self.v.size(0)))

      def score(self, hidden, en_output):
            # batch size, seq len, de embed size*2 + de hidden size -> batch size, seq len, de hidden size
            energy = torch.tanh(self.attn(torch.cat([hidden, en_output], 2)))
            # batch size, de hidden size, seq len
            energy = energy.transpose(2,1)
            # batch size, 1, hidden size
            v = self.v.repeat(en_output.data.shape[0],1).unsqueeze(1)
            # batch size, 1, seq len
            energy = torch.bmm(v,energy)
            # batch size, seq len
            return energy.squeeze(1)

      def forward(self, hidden, en_output, max_src_len, src_len):
            # hidden: (1, batch_size, de hidden size), (1, batch_size, de hidden size)
            # en_output: batch_size, max_seq_len, en_hidden_size*num_directions
            # batch size, de hidden size*2
            hidden = torch.cat(hidden, 2)[-1]
            # batch size, seq len, de hidden0, 1 size*2
            H = hidden.repeat(max_src_len,1,1).transpose(0,1)
            # batch size, seq len
            attn_energies = self.score(H, en_output)
            idx = torch.arange(
                  end=max_src_len, 
                  dtype=torch.float, 
                  device=self.config.device)
            # batch size, seq len
            idx = idx.unsqueeze(0).expand(attn_energies.size())
            # batch size, seq len
            len_expanded = src_len.unsqueeze(-1).expand(attn_energies.size())
            mask = idx < len_expanded
            # batch size, seq len
            attn_energies[~mask] = float('-inf')
            # batch size, 1, seq len
            return torch.nn.functional.softmax(attn_energies, dim=1).unsqueeze(1)

class AttDecoder(torch.nn.Module):
      """docstring for AttDecoder"""
      def __init__(self, config):
            super(AttDecoder, self).__init__()
            self.config = config
            self.embedding = torch.nn.Embedding(
                  num_embeddings=self.config.vocab_size, 
                  embedding_dim=self.config.embedding_size)
            self.em_dropout = torch.nn.Dropout(self.config.embedding_drop_rate)
            self.attn = Attn(self.config)
            self.lstm = torch.nn.LSTM(
                  input_size=self.config.embedding_size, 
                  hidden_size=self.config.de_num_units, 
                  num_layers=self.config.de_num_layers, 
                  batch_first=True, 
                  bidirectional=False)
            self.de_dropout = torch.nn.Dropout(self.config.de_drop_rate)
            self.attn_combine = torch.nn.Linear(
                  self.config.de_num_units+self.config.embedding_size, self.config.de_num_units)
            self.out = torch.nn.Linear(self.config.de_num_units, self.config.vocab_size)
            self.softmax = torch.nn.LogSoftmax(dim=1)

      def forward(self, tar_token, last_hidden, en_output, max_src_len, src_len):
            # tar_token: batch_size
            # last_hidden: (1, batch_size, de hidden size, 1, batch_size, de hidden size)
            # en_output: batch_size, max_seq_len, en_hidden_size*num_directions
            # batch size, 1, embed size
            de_embedded = self.embedding(tar_token).unsqueeze(1)
            # batch size, 1, embed size
            de_embedded = self.em_dropout(de_embedded)
            # batch size, 1, seq len
            attn_w = self.attn(last_hidden, en_output, max_src_len, src_len)
            # batch size, 1, de hidden size
            context = attn_w.bmm(en_output)
            # batch size, 1, de hidden size
            de_input = self.attn_combine(torch.cat((de_embedded, context), 2))
            self.lstm.flatten_parameters()
            de_output, de_hidden = self.lstm(de_input, last_hidden)
            de_output = self.de_dropout(de_output)
            # batch size, de hidden size
            de_output = de_output.squeeze(1)
            # batch size, tar vocab size
            de_output = torch.nn.functional.log_softmax(self.out(de_output), dim=1)
            return de_output, de_hidden

class BeamSearchNode(object):
      """docstring for BeamSearchNode"""
      def __init__(self, hidden, previous_node, token_id, log_prob, length):
            super(BeamSearchNode, self).__init__()
            self.h = hidden
            self.prev_n = previous_node
            self.t_id = token_id
            self.log_p = log_prob
            self.len = length

      def eva(self, alpha=1.0):
            # define reward function here
            reward=0
            return self.log_p/float(self.len-1+1e-6) + alpha*reward