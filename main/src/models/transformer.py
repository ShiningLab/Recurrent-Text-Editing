#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Ziheng Zeng'


# dependency
# public
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class End2EndModelGraph(nn.Module):

    def __init__(self, config):
        super(End2EndModelGraph, self).__init__()
        self.config = config
        # embedding layers
        self.src_embedding_layer = nn.Embedding(config.src_vocab_size, config.embedding_size)
        self.tgt_embedding_layer = nn.Embedding(config.tgt_vocab_size, config.embedding_size)
        # positional encoder
        self.pos_encoder = PositionalEncoding(config.embedding_size, config.pos_encoder_drop_rate)
        # transformer model
        self.transformer_model = nn.Transformer(d_model=config.en_hidden_size,
                                                nhead=config.num_heads,
                                                num_encoder_layers=config.en_num_layers,
                                                num_decoder_layers=config.de_num_layers,
                                                dim_feedforward=config.ffnn_dim,
                                                dropout=config.en_drop_rate)
        # generator layer (a.k.a. the final linear layer)
        self.generator = nn.Linear(config.en_hidden_size, config.tgt_vocab_size)  # encoder hidden dim = decoder hidden dim

    def forward(self, xs, x_lens, ys, src_mask, tgt_mask):
        if self.training:
            # shape: [batch_size, src_seq_len, emb_size]
            xs = self.src_embedding_layer(xs) * math.sqrt(self.config.embedding_size)
            # shape: [src_seq_len, batch_size, emb_size]
            xs = xs.permute(1, 0, 2)
            # shape: [tgt_seq_len, batch_size, emb_size]
            ys = self.tgt_embedding_layer(ys)
            # shape: [tgt_seq_len, batch_size, emb_size]
            ys = ys.permute(1, 0, 2)
            # shape: [src_seq_len, batch_size, emb_size]
            xs = self.pos_encoder(xs)
            # shape: [tgt_seq_len, batch_size, emb_size]
            output = self.transformer_model(xs, ys, src_mask=src_mask, tgt_mask=tgt_mask)
            # shape: [tgt_seq_len, batch_size, tgt_vocab_size]
            output = F.log_softmax(self.generator(output), dim=-1)
            # shape: [batch_size, tgt_seq_len, tgt_vocab_size]
            output = output.permute(1, 0, 2)
            return output
        else:
            max_ys_seq_len = ys.shape[1]
            # shape: [batch_size, src_seq_len, emb_size]
            xs = self.src_embedding_layer(xs) * math.sqrt(self.config.embedding_size)
            xs = self.pos_encoder(xs)
            # shape: [src_seq_len, batch_size, emb_size]
            xs = xs.permute(1, 0, 2)
            # (max_len, batch_size, hidden_dim)
            encoder_hidden_states = self.transformer_model.encoder(xs, src_mask)
            ys = torch.ones(xs.size(1), 1).fill_(self.config.start_idx).type_as(xs.data).long()
            ys_mask = torch.ones(xs.size(1), 1).fill_(1).type_as(xs.data).long()
            end_list = torch.ones(xs.size(1), 1).fill_(self.config.end_idx).type_as(xs.data).long()
            for i in range(max_ys_seq_len - 1):
                # shape: [tgt_seq_len, batch_size, emb_size]
                ys_emb = self.tgt_embedding_layer(ys)
                # shape: [tgt_seq_len, batch_size, emb_size]
                ys_emb = ys_emb.permute(1, 0, 2)
                # (cur_seq_len, batch_size, hidden_dim)
                out = self.transformer_model.decoder(ys_emb,
                                                     encoder_hidden_states,
                                                     tgt_mask=Variable(self.transformer_model.generate_square_subsequent_mask(ys_emb.size(0)).type_as(xs.data)),
                                                     memory_mask=None)
                # (1, batch_size, hidden_dim)
                out = out[-1, :, :].unsqueeze(0)
                # (1, batch_size, vocab_size)
                prob = self.generator(out)
                # (1) (1)
                prob, next_word = torch.max(prob, dim=2)
                next_word = (next_word.permute(1, 0) * ys_mask).type_as(xs.data)
                # process sequences that reach end symbol TODO: make this more general, currently assume the padding idx is 0!
                match_end_idx = (next_word.long() == end_list).type_as(xs.data)
                mask_flip = match_end_idx.clone()
                mask_flip[match_end_idx == 0] = 1
                mask_flip[match_end_idx != 0] = 0
                ys_mask = ys_mask * mask_flip.long()
                    # (cur_seq_len)
                ys = torch.cat([ys.long(), next_word.data.long()], dim=1)

            return ys


class RecursionModelGraph(nn.Module):

    def __init__(self, config):
        super(RecursionModelGraph, self).__init__()
        self.config = config
        # embedding layers
        self.src_embedding_layer = nn.Embedding(config.src_vocab_size, config.embedding_size)
        self.tgt_embedding_layer = nn.Embedding(config.tgt_vocab_size, config.embedding_size)
        # positional encoder
        self.pos_encoder = PositionalEncoding(config.embedding_size, config.pos_encoder_drop_rate)
        # transformer model
        self.transformer_model = nn.Transformer(d_model=config.en_hidden_size,
                                                nhead=config.num_heads,
                                                num_encoder_layers=config.en_num_layers,
                                                num_decoder_layers=config.de_num_layers,
                                                dim_feedforward=config.ffnn_dim,
                                                dropout=config.en_drop_rate)
        # generator layer (a.k.a. the final linear layer)
        self.generator = nn.Linear(config.en_hidden_size, config.tgt_vocab_size)  # encoder hidden dim = decoder hidden dim

    def forward(self, xs, x_lens, ys, src_mask, tgt_mask):
        if self.training:
            # shape: [batch_size, src_seq_len, emb_size]
            xs = self.src_embedding_layer(xs) * math.sqrt(self.config.embedding_size)
            # shape: [src_seq_len, batch_size, emb_size]
            xs = xs.permute(1, 0, 2)
            # shape: [tgt_seq_len, batch_size, emb_size]
            ys = self.tgt_embedding_layer(ys)
            # shape: [tgt_seq_len, batch_size, emb_size]
            ys = ys.permute(1, 0, 2)
            # shape: [src_seq_len, batch_size, emb_size]
            xs = self.pos_encoder(xs)
            # shape: [tgt_seq_len, batch_size, emb_size]
            output = self.transformer_model(xs, ys, src_mask=src_mask, tgt_mask=tgt_mask)
            # shape: [tgt_seq_len, batch_size, tgt_vocab_size]
            output = F.log_softmax(self.generator(output), dim=-1)
            # shape: [batch_size, tgt_seq_len, tgt_vocab_size]
            output = output.permute(1, 0, 2)
            return output
        else:
            max_ys_seq_len = self.config.tgt_seq_len
            # shape: [batch_size, src_seq_len, emb_size]
            xs = self.src_embedding_layer(xs) * math.sqrt(self.config.embedding_size)
            xs = self.pos_encoder(xs)
            # shape: [src_seq_len, batch_size, emb_size]
            xs = xs.permute(1, 0, 2)
            # (max_len, batch_size, hidden_dim)
            encoder_hidden_states = self.transformer_model.encoder(xs, src_mask)
            ys = torch.ones(xs.size(1), 1).fill_(self.config.start_idx).type_as(xs.data).long()
            ys_mask = torch.ones(xs.size(1), 1).fill_(1).type_as(xs.data).long()
            # end_list = torch.ones(xs.size(1), 1).fill_(self.config.end_idx).type_as(xs.data).long()
            for i in range(max_ys_seq_len- 1):
                # shape: [tgt_seq_len, batch_size, emb_size]
                ys_emb = self.tgt_embedding_layer(ys)
                # shape: [tgt_seq_len, batch_size, emb_size]
                ys_emb = ys_emb.permute(1, 0, 2)
                # (cur_seq_len, batch_size, hidden_dim)
                out = self.transformer_model.decoder(ys_emb,
                                                     encoder_hidden_states,
                                                     tgt_mask=Variable(self.transformer_model.generate_square_subsequent_mask(ys_emb.size(0)).type_as(xs.data)),
                                                     memory_mask=None)
                # (1, batch_size, hidden_dim)
                out = out[-1, :, :].unsqueeze(0)
                # (1, batch_size, vocab_size)
                prob = self.generator(out)
                # (1) (1)
                prob, next_word = torch.max(prob, dim=2)
                next_word = (next_word.permute(1, 0) * ys_mask).type_as(xs.data)
                # process sequences that reach end symbol TODO: make this more general, currently assume the padding idx is 0!
                # match_end_idx = (next_word.long() == end_list).type_as(xs.data)
                # mask_flip = match_end_idx.clone()
                # mask_flip[match_end_idx == 0] = 1
                # mask_flip[match_end_idx != 0] = 0
                # ys_mask = ys_mask * mask_flip.long()
                # (cur_seq_len)
                ys = torch.cat([ys.long(), next_word.data.long()], dim=1)

            return ys


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

