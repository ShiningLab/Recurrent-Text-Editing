#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'

# public
import torch
from torch.utils import data as torch_data

import os
from tqdm import tqdm

# private
from config import Config
from src.utils.eva import Evaluate
from src.utils.save import *
from src.utils.load import *
from src.utils.pipeline import *


class TextEditor(object):
    """docstring for TextEditor"""
    def __init__(self, config):
        super(TextEditor, self).__init__()
        self.step, self.epoch = 0, 0 # training step and epoch
        self.finished = False # training done flag
        self.valid_step, self.valid_epoch = 0, 0  # validation step and epoch
        self.valid_loss, self.valid_acc = float('-inf'), float('-inf')
        self.valid_token_acc, self.valid_seq_acc = float('-inf'), float('-inf')
        self.test_log = []
        self.config = config
        self.setup_gpu()
        self.load_vocab()
        self.load_data()
        self.setup_model()

    def setup_gpu(self): 
        # verify devices which can be either cpu or gpu
        self.config.use_gpu = torch.cuda.is_available()
        self.config.device = 'cuda' if self.config.use_gpu else 'cpu'

    def load_vocab(self):
        # load the vocab dictionary and update config
        self.vocab2idx_dict = load_json(self.config.VOCAB_PATH)
        self.idx2vocab_dict = {v: k for k, v in self.vocab2idx_dict.items()}
        self.config.pad_idx = self.vocab2idx_dict[self.config.pad_symbol]
        self.config.start_idx = self.vocab2idx_dict[self.config.start_symbol]
        self.config.end_idx = self.vocab2idx_dict[self.config.end_symbol]
        self.config.vocab_size = len(self.vocab2idx_dict)

    def end2end_collate_fn(self, data): 
        # a customized collate function used in the data loader 
        def preprocess(xs, ys): 
            # add start and end symbol 
            xs = [torch.Tensor(x + [self.config.end_idx]) for x in xs]
            ys = [torch.Tensor(y + [self.config.end_idx]) for y in ys] 
            return xs, ys

        def padding(seqs):
            # zero padding
            seq_lens = [len(seq) for seq in seqs]
            padded_seqs = torch.zeros([len(seqs), max(seq_lens)], dtype=torch.int64)
            for i, seq in enumerate(seqs): 
                seq_len = seq_lens[i]
                padded_seqs[i, :seq_len] = seq[:seq_len]
            return padded_seqs

        data.sort(key=len, reverse=True)
        xs, ys = zip(*data)
        xs, ys = preprocess(xs, ys)
        xs = padding(xs)
        ys = padding(ys)

        return xs.to(self.config.device), ys.to(self.config.device)

    def load_data(self): 
        # read data dictionary from json file
        self.data_dict = load_json(self.config.DATA_PATH)
        # train data loader
        self.train_dataset = Een2EndDataset(self.data_dict['train'])
        self.trainset_generator = torch_data.DataLoader(
              self.train_dataset, 
              batch_size=self.config.batch_size, 
              collate_fn=self.end2end_collate_fn, 
              shuffle=self.config.shuffle, 
              drop_last=self.config.drop_last)
        # valid data loader
        self.valid_dataset = Een2EndDataset(self.data_dict['valid'])
        self.validset_generator = torch_data.DataLoader(
              self.valid_dataset, 
              batch_size=self.config.batch_size, 
              collate_fn=self.end2end_collate_fn, 
              shuffle=False, 
              drop_last=False)
        # test data loader
        self.test_dataset = Een2EndDataset(self.data_dict['test'])
        self.testset_generator = torch_data.DataLoader(
              self.test_dataset, 
              batch_size=self.config.batch_size, 
              collate_fn=self.end2end_collate_fn, 
              shuffle=False,
              drop_last=False)
        self.config.train_size = len(self.train_dataset)
        self.config.train_batch = len(self.trainset_generator)
        self.config.valid_size = len(self.valid_dataset)
        self.config.valid_batch = len(self.validset_generator)
        self.config.test_size = len(self.test_dataset)
        self.config.test_batch = len(self.testset_generator)
        self.check_iteration = [int(self.config.train_batch*i) for i in [0.25, 0.5, 0.75, 1.]]

    def load_check_point(self):
        checkpoint_to_load =  torch.load(self.config.LOAD_POINT, map_location=self.config.device) 
        self.step = checkpoint_to_load['step'] 
        self.epoch = checkpoint_to_load['epoch'] 
        model_state_dict = checkpoint_to_load['model'] 
        self.model.load_state_dict(model_state_dict) 
        self.opt.load_state_dict(checkpoint_to_load['optimizer'])

    def setup_model(self): 
        # initialize model weights, optimizer, and loss function
        self.model = pick_model('gru_rnn', self.config)
        self.model.apply(init_parameters)
        self.criterion = torch.nn.NLLLoss(ignore_index=self.config.pad_idx)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        if self.config.load_check_point: 
            self.load_check_point()
        self.config.num_parameters = count_parameters(self.model)

    def train(self):
        show_config(self.config, self.model)
        while not self.finished:
            print('\nTraining...')
            self.model.train()
            # training set data loader
            trainset_generator = tqdm(self.trainset_generator)
            for i, (xs, ys) in enumerate(trainset_generator): 
                ys_ = self.model(xs, ys, self.config.teacher_forcing_ratio)
                loss = self.criterion(ys_.reshape(-1, self.config.vocab_size), ys.reshape(-1))
                # update step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clipping_threshold)
                self.opt.step()
                self.opt.zero_grad()
                # check progress
                if i+1 in self.check_iteration:
                    # post processing
                    loss = loss.item()
                    xs = xs.cpu().detach().numpy() # batch_size, max_xs_seq_len
                    ys = ys.cpu().detach().numpy() # batch_size, max_ys_seq_len
                    ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy() # batch_size, max_ys_seq_len
                    xs, ys, ys_ = prepare_output(xs, ys, ys_, self.config.pad_idx)
                    # evaluation
                    eva_matrix = Evaluate(self.config, ys, ys_, self.idx2vocab_dict)
                    # update processing bar
                    trainset_generator.set_description(
                        'Train Epoch {} Total Step {} Loss:{:.4f} Acc:{:.4f} Token Acc:{:.4f} Seq Acc:{:.4f}'.format(
                        self.epoch, self.step, loss, eva_matrix.acc, eva_matrix.token_acc, eva_matrix.seq_acc))
                    trainset_generator.refresh()
                    # random sample to show
                    src, tar, pred = rand_sample(xs, ys, ys_, self.idx2vocab_dict)
                    print(' src: {}\n tar: {}\n pred: {}'.format(src, tar, pred))
                
                self.step += 1

            self.validate()
            self.test()
            save_txt(self.config.LOG_PATH, self.test_log)
            if self.epoch >= self.config.test_epoch: 
                self.finished = True
                # save_txt()

            self.epoch += 1

    def validate(self):
        print('\nValidating...')
        all_loss = 0
        all_xs, all_ys, all_ys_ = [], [], []
        validset_generator = tqdm(self.validset_generator)
        self.model.eval()
        with torch.no_grad(): 
            for xs, ys in validset_generator: 
                ys_ = self.model(xs, ys, teacher_forcing_ratio=0.)
                loss = self.criterion(ys_.reshape(-1, self.config.vocab_size), ys.reshape(-1))
                loss = loss.item()
                xs = xs.cpu().detach().numpy() # batch_size, max_xs_seq_len
                ys = ys.cpu().detach().numpy() # batch_size, max_ys_seq_len
                ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy() # batch_size, max_ys_seq_len
                xs, ys, ys_ = prepare_output(xs, ys, ys_, self.config.pad_idx)
                all_loss += loss
                all_xs += xs
                all_ys += ys 
                all_ys_ += ys_

        all_loss /= self.config.valid_batch
        eva_matrix = Evaluate(self.config, all_ys, all_ys_, self.idx2vocab_dict)
        print('Validation Epoch:{} Total Step:{} Loss:{:.4f} Acc:{:.4f} Token Acc:{:.4f} Seq Acc:{:.4f}'.format(
            self.epoch, self.step, all_loss, eva_matrix.acc, eva_matrix.token_acc, eva_matrix.seq_acc))
        # save model on the basis of sequence accuracy
        if eva_matrix.seq_acc >= self.valid_seq_acc:
            self.valid_epoch, self.valid_step = self.epoch, self.step
            self.valid_loss = all_loss
            self.valid_acc = eva_matrix.acc
            self.valid_token_acc = eva_matrix.token_acc
            self.valid_seq_acc = eva_matrix.seq_acc
            # save model
            save_path = self.config.SAVE_POINT.format(self.step, self.epoch)
            save_check_point(self.step, self.epoch, 
                self.model.state_dict, self.opt.state_dict, save_path)

        src, tar, pred = rand_sample(all_xs, all_ys, all_ys_, self.idx2vocab_dict)
        print(' src: {}\n tar: {}\n pred: {}'.format(src, tar, pred))

        print('Best Validation Epoch:{} Step:{} Loss:{:.4f} Acc:{:.4f} Token Acc:{:.4f} Seq Acc:{:.4f}'.format(
            self.valid_epoch, self.valid_step, self.valid_loss, self.valid_acc, self.valid_token_acc, self.valid_seq_acc))

    def test(self):
        print('\nTesting...')
        all_loss = 0
        all_xs, all_ys, all_ys_ = [], [], []
        # initialize the model and load best check point
        model = pick_model('gru_rnn', self.config)

        load_points = [p for p in os.listdir(self.config.SAVE_PATH) if p.endswith('.pt')]
        load_point = sorted(load_points, key=lambda x: int(x.split('_')[0]))[-1] 
        load_point = os.path.join(self.config.SAVE_PATH, load_point)
        checkpoint_to_load =  torch.load(load_point, map_location=self.config.device) 
        step = checkpoint_to_load['step'] 
        epoch = checkpoint_to_load['epoch'] 
        model_state_dict = checkpoint_to_load['model'] 
        model.load_state_dict(model_state_dict) 
        print('Model restored from {}.'.format(self.config.LOAD_POINT))
        # star testing
        testset_generator = tqdm(self.testset_generator)
        model.eval()
        with torch.no_grad():
            for xs, ys in testset_generator:
                ys_ = model(xs, ys, teacher_forcing_ratio=0.)
                loss = self.criterion(ys_.reshape(-1, self.config.vocab_size), ys.reshape(-1))
                loss = loss.item()
                xs = xs.cpu().detach().numpy() # batch_size, max_xs_seq_len
                ys = ys.cpu().detach().numpy() # batch_size, max_ys_seq_len
                ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy() # batch_size, max_ys_seq_len
                xs, ys, ys_ = prepare_output(xs, ys, ys_, self.config.pad_idx)
                all_loss += loss
                all_xs += xs
                all_ys += ys 
                all_ys_ += ys_

        all_loss /= self.config.test_batch
        eva_matrix = Evaluate(self.config, all_ys, all_ys_, self.idx2vocab_dict)
        log_msg = 'Test Epoch:{} Total Step:{} Loss:{:.4f} Acc:{:.4f} Token Acc:{:.4f} Seq Acc:{:.4f}'.format(
            epoch, step, all_loss, eva_matrix.acc, eva_matrix.token_acc, eva_matrix.seq_acc)
        self.test_log.append(log_msg)
        print(log_msg)

def main(): 
    # initial everything
    te = TextEditor(Config())
    te.train()

if __name__ == '__main__':
      main()