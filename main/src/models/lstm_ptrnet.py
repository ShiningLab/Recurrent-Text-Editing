__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'

# dependency
# public
import torch
import torch.nn as nn
import random
import torch.nn.functional as F

# private
from .encoder import BiLSTMRNNEncoder
# from .decoder import LSTMPtrNetDecoder


class PtrNetAttention(nn.Module):
    """docstring for PtrNetAttention"""
    def __init__(self, config):
        super(PtrNetAttention, self).__init__()
        self.config = config
        self.w1 = nn.Linear(config.en_hidden_size, config.de_hidden_size, bias=False)
        self.w2 = nn.Linear(config.en_hidden_size, config.de_hidden_size, bias=False)
        self.vt = nn.Linear(config.de_hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_output, src_lens):
        # encoder_output: batch_size, max_xs_seq_len, en_hidden_size
        # decoder_hidden: (h, c)
        # h, c: 1, batch_size, en_hidden_size
        # shape: [batch_size, 1, en_hidden_size]
        hs = decoder_hidden[0][0].unsqueeze(1)  # take only the hidden state

        # compute attention
        # shape: [batch_size, max_seq_len, en_hidden_size]
        uj = self.w1(encoder_output) + self.w2(hs)
        uj = torch.tanh(uj)
        # shape: [batch_size, max_seq_len, 1]
        uj = self.vt(uj)

        # apply attention to the encoder output to get context vector
        # shape: [batch_size, max_seq_len, 1]
        aj = F.softmax(uj, dim=1)
        # shape: [batch_size, max_seq_len, en_hidden_size]
        context = aj * encoder_output
        # shape: [batch_size, en_hidden_size]
        context = context.sum(1)

        return context, uj.squeeze(-1)


class LSTMPtrNetDecoder(nn.Module):
    """docstring for LSTMPtrNetDecoder"""
    def __init__(self, config):
        super(LSTMPtrNetDecoder, self).__init__()
        self.config = config
        self.attention_layer = PtrNetAttention(config)
        self.lstm = nn.LSTM(
            input_size=self.config.de_hidden_size + 1,
            hidden_size=self.config.de_hidden_size,
            num_layers=self.config.de_num_layers,
            batch_first=True,
            dropout=0,
            bidirectional=False)

    def forward(self, x, decoder_hidden, encoder_output, src_lens):
        # shape: context: [batch_size, en_hidden_size] | attn: [batch_size, max_seq_len]
        context, attn = self.attention_layer(decoder_hidden, encoder_output, src_lens)

        # apply context to the input
        # shape: [batch_size, hidden_size]
        context = context.unsqueeze(1)
        # shape: [batch_size, 1, hidden_size+1]
        x = torch.cat([context, x], dim=2)
        # shape: 2 x [1, batch_size, hidden_size]
        _, decoder_hidden = self.lstm(x, decoder_hidden)

        return decoder_hidden, attn


class End2EndModelGraph(nn.Module):
    """docstring for End2EndModelGraph"""
    def __init__(self, config):
        super(End2EndModelGraph, self).__init__()
        self.config = config
        self.encoder = nn.LSTM(1, self.config.en_hidden_size, batch_first=True)
        self.decoder = LSTMPtrNetDecoder(config)
        self.embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.config.embedding_size,
            padding_idx=self.config.pad_idx)

    def forward(self, xs, x_lens, argsort_xs, teacher_forcing_ratio=0.5):
        # xs: batch_size, max_len
        # argsoft_xs: batch_size, max_len
        batch_size = xs.shape[0]
        # for ptr net, x_len = y_len = max_len
        max_len = xs.shape[1]
        # encoder_output: batch_size, max_xs_seq_len, en_hidden_size
        # decoder_hidden: (h, c)
        # h, c: 1, batch_size, en_hidden_size
        encoder_output, decoder_hidden = self.encoder(xs.unsqueeze(-1).type(torch.float))

        # Create decoder output place holder
        # decoder_output = torch.zeros(encoder_output.size(1), encoder_output.size(0), dtype=torch.long)
        # max_len, batch_size, max_len
        decoder_outputs = torch.zeros(max_len, batch_size, max_len, device=self.config.device)

        # Create decoder input
        # shape: (batch_size, 1, 1)
        decoder_input = torch.zeros(encoder_output.size(0), 1, 1, dtype=torch.float, device=self.config.device)

        for i in range(max_len):
            # shape: 2 x [1, batch_size, hidden_size] | [batch_size, max_seq_len]
            decoder_hidden, attn = self.decoder(decoder_input, decoder_hidden, encoder_output, x_lens)
            # shape: [batch_size]
            next_tokens = F.softmax(attn, dim=1).argmax(1)
            # select the next tokens
            # shape: [batch_size]
            ptr_indices = argsort_xs[:, i, None] if random.random() < teacher_forcing_ratio else next_tokens
            # shape: [batch_size]
            decoder_input = torch.stack([xs[t, ptr_indices[t].item()] for t in range(batch_size)])
            # shape: [batch_size, 1, 1]
            decoder_input = decoder_input.view(batch_size, 1, 1).type(torch.float)
            # shape: [max_seq_len, batch_size, max_seq_len]
            decoder_outputs[i] = F.log_softmax(attn)

        return decoder_outputs.permute(1, 0, 2)