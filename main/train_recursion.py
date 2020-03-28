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
from config import RecursionConfig
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
        vocab_dict = load_json(self.config.VOCAB_PATH)
        self.src_vocab2idx_dict = vocab_dict['src']
        self.tgt_vocab2idx_dict = vocab_dict['tgt']
        self.src_idx2vocab_dict = {v: k for k, v in self.src_vocab2idx_dict.items()}
        self.tgt_idx2vocab_dict = {v: k for k, v in self.tgt_vocab2idx_dict.items()}
        self.config.pad_idx = self.src_vocab2idx_dict[self.config.pad_symbol]
        self.config.start_idx = self.src_vocab2idx_dict[self.config.start_symbol]
        self.config.end_idx = self.src_vocab2idx_dict[self.config.end_symbol]
        self.config.src_vocab_size = len(self.src_vocab2idx_dict)
        self.config.tgt_vocab_size = len(self.tgt_vocab2idx_dict)

    def offline_recursion_collate_fn(self, data): 
        # a customized collate function used in the data loader 
        xs, ys = zip(*data)
        xs, ys = preprocess(
            xs, ys, self.src_vocab2idx_dict, self.src_vocab2idx_dict, self.config.end_idx)
        xs, x_lens = padding(xs)
        ys, _ = padding(ys)

        return (xs.to(self.config.device), 
            torch.Tensor(x_lens).to(self.config.device), 
            ys.to(self.config.device))

    def online_recursion_collate_fn(self, data): 
        # a customized collate function used in the data loader 
        xs, ys = zip(*[get_sequence_pair(y) for y in data])
        xs, ys = preprocess(
            xs, ys, self.src_vocab2idx_dict, self.tgt_vocab2idx_dict, self.config.end_idx)
        xs, x_lens = padding(xs)
        ys, _ = padding(ys)

        return (xs.to(self.config.device), 
            torch.Tensor(x_lens).to(self.config.device), 
            ys.to(self.config.device))

    def test_recursion_collate_fn(self, data): 
        # a customized collate function used in the data loader 
        xs, ys = zip(*data)
        xs, ys = preprocess(
            xs, ys, self.src_vocab2idx_dict, self.src_vocab2idx_dict, self.config.end_idx)
        xs, x_lens = padding(xs, self.config.seq_len*2)
        ys, _ = padding(ys)

        return (xs.to(self.config.device), 
            torch.Tensor(x_lens).to(self.config.device), 
            ys.to(self.config.device))

    def load_data(self): 
        # read data dictionary from json file
        self.data_dict = load_json(self.config.DATA_PATH)
        # train data loader
        self.train_dataset = OnlineRecursionDataset(self.data_dict['train'])
        self.trainset_generator = torch_data.DataLoader(
              self.train_dataset, 
              batch_size=self.config.batch_size, 
              collate_fn=self.online_recursion_collate_fn, 
              shuffle=self.config.shuffle, 
              drop_last=self.config.drop_last)
        # valid data loader
        self.valid_dataset = OnlineRecursionDataset(self.data_dict['test'])
        self.validset_generator = torch_data.DataLoader(
            self.valid_dataset, 
            batch_size=self.config.batch_size, 
            collate_fn=self.online_recursion_collate_fn, 
            shuffle=False, 
            drop_last=False)
        # test data loader
        self.test_dataset = Dataset(self.data_dict['test'])
        self.testset_generator = torch_data.DataLoader(
              self.test_dataset, 
              batch_size=self.config.batch_size, 
              collate_fn=self.test_recursion_collate_fn, 
              shuffle=False,
              drop_last=False)
        self.config.train_size = len(self.train_dataset)
        self.config.train_batch = len(self.trainset_generator)
        self.config.test_size = len(self.test_dataset)
        self.config.test_batch = len(self.testset_generator)

    def load_check_point(self):
        checkpoint_to_load =  torch.load(self.config.LOAD_POINT, map_location=self.config.device) 
        self.step = checkpoint_to_load['step'] 
        self.epoch = checkpoint_to_load['epoch'] 
        model_state_dict = checkpoint_to_load['model'] 
        self.model.load_state_dict(model_state_dict) 
        self.opt.load_state_dict(checkpoint_to_load['optimizer'])

    def setup_model(self): 
        # initialize model weights, optimizer, and loss function
        self.model = pick_model(self.config, 'end2end')
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
            for i, (xs, x_lens, ys) in enumerate(trainset_generator): 
                # print(x_lens.cpu().detach().numpy()[0])
                # print(translate(xs.cpu().detach().numpy()[0], self.src_idx2vocab_dict))
                # print(translate(ys.cpu().detach().numpy()[0], self.tgt_idx2vocab_dict))
                # break
            # break
                ys_ = self.model(xs, x_lens, ys, self.config.teacher_forcing_ratio)
                loss = self.criterion(ys_.reshape(-1, self.config.tgt_vocab_size), ys.reshape(-1))
                # update step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clipping_threshold)
                self.opt.step()
                self.opt.zero_grad()
                self.step += 1
            # check progress
            loss = loss.item()
            xs = xs.cpu().detach().numpy() # batch_size, max_xs_seq_len
            ys = ys.cpu().detach().numpy() # batch_size, max_ys_seq_len
            ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy() # batch_size, max_ys_seq_len
            xs, ys, ys_ = prepare_output(xs, ys, ys_, self.config.pad_idx)
            # evaluation
            eva_matrix = Evaluate(self.config, ys, ys_, self.tgt_idx2vocab_dict)
            print('Train Epoch {} Total Step {} Loss:{:.4f} Token Acc:{:.4f} Seq Acc:{:.4f}'.format(
                self.epoch, self.step, loss, eva_matrix.token_acc, eva_matrix.seq_acc))
            # random sample to show
            src, tar, pred = rand_sample(xs, ys, ys_, 
                self.src_idx2vocab_dict, self.tgt_idx2vocab_dict, self.tgt_idx2vocab_dict)
            print(' src: {}\n tar: {}\n pred: {}'.format(src, tar, pred))
            # validation
            self.valid()
            # testing
            self.test()
            # if done
            if self.epoch >= self.config.test_epoch: 
                # update flag
                self.finished = True
                # save log
                save_txt(self.config.LOG_PATH, self.test_log)
                # save model
                save_check_point(self.step, self.epoch, self.model.state_dict, self.opt.state_dict, self.config.SAVE_POINT)

            self.epoch += 1

    def valid(self):
        print('\nValidating...')
        all_loss = 0
        all_xs, all_ys, all_ys_ = [], [], []
        validset_generator = tqdm(self.validset_generator)
        self.model.eval()
        with torch.no_grad():
            for xs, x_lens, ys in validset_generator:
                ys_ = self.model(xs, x_lens, ys, teacher_forcing_ratio=0.)
                loss = self.criterion(ys_.reshape(-1, self.config.tgt_vocab_size), ys.reshape(-1))
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
        eva_matrix = Evaluate(self.config, all_ys, all_ys_, self.tgt_idx2vocab_dict)
        print('Valid Epoch:{} Total Step:{} Loss:{:.4f} Token Acc:{:.4f} Seq Acc:{:.4f}'.format(
            self.epoch, self.step, all_loss, eva_matrix.token_acc, eva_matrix.seq_acc))
        # random sample to show
        src, tar, pred = rand_sample(xs, ys, ys_, 
            self.src_idx2vocab_dict, self.tgt_idx2vocab_dict, self.tgt_idx2vocab_dict)
        print(' src: {}\n tar: {}\n pred: {}'.format(src, tar, pred))

    def test(self):
        print('\nTesting...')
        model = pick_model(self.config, 'recursion')
        # checkpoint_to_load =  torch.load(self.config.LOAD_POINT, map_location=self.config.device) 
        # print('Model restored from {}.'.format(self.config.LOAD_POINT))
        # step = checkpoint_to_load['step'] 
        # epoch = checkpoint_to_load['epoch'] 
        # model_state_dict = checkpoint_to_load['model'] 
        # model.load_state_dict(model_state_dict) 
        model.load_state_dict(self.model.state_dict())
        all_xs, all_ys = [], [] 
        # all_ys_ = []
        # trainset_generator = tqdm(self.trainset_generator)
        # validset_generator = tqdm(self.validset_generator)
        testset_generator = tqdm(self.testset_generator)
        model.eval()
        with torch.no_grad():
            for xs, x_lens, ys in testset_generator: 
                for i in range(self.config.seq_len):
                    # print(x_lens.cpu().detach().numpy()[0])
                    # print(translate(xs.cpu().detach().numpy()[0], self.src_idx2vocab_dict))
                    # print(translate(ys.cpu().detach().numpy()[0], self.src_idx2vocab_dict))
                    ys_ = model(xs, x_lens)
                    xs, x_lens = recursive_infer(xs, x_lens, ys_, 
                        self.src_idx2vocab_dict, self.src_vocab2idx_dict, 
                        self.tgt_idx2vocab_dict, self.config)

                xs = xs.cpu().detach().numpy() # batch_size, max_xs_seq_len
                ys = ys.cpu().detach().numpy() # batch_size, max_ys_seq_len

                # ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy() # batch_size, max_ys_seq_len
                # xs, ys, ys_ = prepare_output(xs, ys, ys_, self.config.pad_idx)

                xs = [rm_pad(x, self.config.pad_idx) for x in xs] 
                ys = [rm_pad(y, self.config.pad_idx) for y in ys] 

                all_xs += xs
                all_ys += ys 
                # all_ys_ += ys_

        eva_matrix = Evaluate(self.config, all_ys, all_xs, self.src_idx2vocab_dict)
        log_msg = 'Test Epoch:{} Total Step:{} Acc:{:.4f} Token Acc:{:.4f} Seq Acc:{:.4f}'.format(
            self.epoch, self.step, eva_matrix.acc, eva_matrix.token_acc, eva_matrix.seq_acc)
        self.test_log.append(log_msg)
        print(log_msg)

        # vocab to index
        # all_xs = [translate(x, self.src_idx2vocab_dict) for x in all_xs]
        # all_ys = [translate(y, self.src_idx2vocab_dict) for y in all_ys]
        # all_ys_ = [translate(y, self.tgt_idx2vocab_dict) for y in all_ys_]

        # for i in range(20):
        #     print('x:', all_xs[i])
        #     print('y:', all_ys[i])
        #     print('y_', all_ys_[i])
        #     print()

def main(): 
    # initial everything
    te = TextEditor(RecursionConfig())
    # train!
    te.train()
    # te.test()

if __name__ == '__main__':
      main()