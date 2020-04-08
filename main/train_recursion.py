#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'

# public
import torch
from torch.utils import data as torch_data

import os
from tqdm import tqdm
from datetime import datetime

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
        self.start_time = datetime.now()
        self.val_key_metric = float('-inf')
        self.val_log = ['Start Time: {}'.format(self.start_time)]
        self.test_log = self.val_log.copy()
        self.config = config
        self.step, self.epoch = 0, 0 # training step and epoch
        self.finished = False # training done flag
        # equation accuracy
        self.val_metric_list =  [float('-inf')]*self.config.val_win_size
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
        self.config.start_idx = self.tgt_vocab2idx_dict[self.config.start_symbol]
        self.config.src_vocab_size = len(self.src_vocab2idx_dict)
        self.config.tgt_vocab_size = len(self.tgt_vocab2idx_dict)

    def train_recursion_collate_fn(self, data):
        # a customized collate function used in the data loader 
        data.sort(key=len, reverse=True)
        if self.config.data_mode == 'online':
            xs, ys = zip(*[recursion_online_generator(self.config.data_src, d) for d in data])
        else:
            xs, ys = zip(*data)
        # convert to index, add end symbol, and save as tensor
        xs, ys = preprocess(
            xs, ys, self.src_vocab2idx_dict, self.tgt_vocab2idx_dict, self.config)
        xs, x_lens = padding(xs)
        ys, _ = padding(ys)

        if self.config.model_name == 'transformer':
            src_mask = prepare_masks(xs.size(1), self.config)
            tgt_mask = prepare_masks(ys.size(1)+1, self.config)  # plus 1 here for the start idx
            return (xs.to(self.config.device), 
                torch.Tensor(x_lens).to(self.config.device), 
                ys.to(self.config.device), 
                src_mask.to(self.config.device), 
                tgt_mask.to(self.config.device))
        else: 
            return (xs.to(self.config.device), 
                torch.Tensor(x_lens).to(self.config.device), 
                ys.to(self.config.device))

    def test_recursion_collate_fn(self, data): 
        # a customized collate function used in the data loader 
        data.sort(key=len, reverse=True)
        xs, ys = zip(*data)
        # convert to index, add end symbol, and save as tensor
        xs, ys = preprocess(
            xs, ys, self.src_vocab2idx_dict, self.src_vocab2idx_dict, self.config)
        # TODO: why padding leads to an incorrect prediction
        xs, x_lens = padding(xs, self.config.seq_len*2)
        # xs, x_lens = padding(xs)
        ys, _ = padding(ys)

        if self.config.model_name == 'transformer':
            src_mask = prepare_masks(xs.size(1), self.config)
            tgt_mask = prepare_masks(ys.size(1)+1, self.config)  # plus 1 here for the start idx
            return (xs.to(self.config.device), 
                torch.Tensor(x_lens).to(self.config.device), 
                ys.to(self.config.device), 
                src_mask.to(self.config.device), 
                tgt_mask.to(self.config.device))
        else:
            return (xs.to(self.config.device), 
                torch.Tensor(x_lens).to(self.config.device), 
                ys.to(self.config.device))

    def load_data(self): 
        # read data dictionary from json file
        self.data_dict = load_json(self.config.DATA_PATH)
        # train data loader
        if self.config.data_mode == 'online':
            self.train_dataset = OnlineRecursionDataset(
                data_dict=self.data_dict['train'], 
                data_src=self.config.data_src)
        else:
            self.train_dataset = OfflineRecursionDataset(
                data_dict=self.data_dict['train'])
        self.trainset_generator = torch_data.DataLoader(
              self.train_dataset, 
              batch_size=self.config.batch_size, 
              collate_fn=self.train_recursion_collate_fn, 
              shuffle=self.config.shuffle, 
              drop_last=self.config.drop_last)
        # valid data loader
        self.val_dataset = OfflineEnd2EndDataset(self.data_dict['val'])
        self.valset_generator = torch_data.DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            collate_fn=self.test_recursion_collate_fn, 
            shuffle=False, 
            drop_last=False)
        # test data loader
        self.test_dataset = OfflineEnd2EndDataset(self.data_dict['test'])
        self.testset_generator = torch_data.DataLoader(
              self.test_dataset, 
              batch_size=self.config.batch_size, 
              collate_fn=self.test_recursion_collate_fn, 
              shuffle=False,
              drop_last=False)
        # update config
        self.config.train_size = len(self.train_dataset)
        self.config.train_batch = len(self.trainset_generator)
        self.config.val_size = len(self.val_dataset)
        self.config.val_batch = len(self.valset_generator)
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
        general_info = show_config(self.config, self.model)
        self.val_log.append(general_info)
        self.test_log.append(general_info)
        while not self.finished:
            print('\nTraining...')
            self.model.train()
            # training set data loader
            trainset_generator = tqdm(self.trainset_generator)
            for i,  data in enumerate(trainset_generator): 
                if self.config.model_name == 'transformer':
                    xs, x_lens, ys, src_mask, tgt_mask = data
                    start_symbols = torch.ones([ys.size(0), 1], 
                        dtype=torch.double, device=self.config.device)*self.config.start_idx
                    ys = torch.cat((start_symbols.long(), ys), 1)
                    ys_ = self.model(xs, x_lens, ys, src_mask, tgt_mask)
                else:
                    xs, x_lens, ys = data
                    ys_ = self.model(xs, x_lens, ys, self.config.teacher_forcing_ratio)
                # print(x_lens.cpu().detach().numpy()[0])
                # print(translate(xs.cpu().detach().numpy()[0], self.src_idx2vocab_dict))
                # print(translate(ys.cpu().detach().numpy()[0], self.tgt_idx2vocab_dict))
                # print(translate(torch.argmax(ys_, dim=2).cpu().detach().numpy()[0], self.tgt_idx2vocab_dict))
            #     break
            # break
                loss = self.criterion(ys_.reshape(-1, self.config.tgt_vocab_size), ys.reshape(-1))
                # print(loss.item())
                # update step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clipping_threshold)
                self.opt.step()
                self.opt.zero_grad()
                self.step += 1
            #     break
            # break
            # check progress
            loss = loss.item()
            xs = xs.cpu().detach().numpy() # batch_size, max_xs_seq_len
            ys = ys.cpu().detach().numpy() # batch_size, max_ys_seq_len
            ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy() # batch_size, max_ys_seq_len
            xs, ys, ys_ = rm_pads(xs, ys, ys_, self.config.pad_idx)
            # evaluation
            eva_matrix = Evaluate(self.config, ys, ys_, self.tgt_idx2vocab_dict, True)
            eva_msg = 'Train Epoch {} Total Step {} Loss:{:.4f} '.format(self.epoch, self.step, loss)
            eva_msg += eva_matrix.eva_msg
            print(eva_msg)
            # random sample to show
            src, tar, pred = rand_sample(xs, ys, ys_, 
                self.src_idx2vocab_dict, self.tgt_idx2vocab_dict, self.tgt_idx2vocab_dict)
            print(' src: {}\n tar: {}\n pred: {}'.format(src, tar, pred))
            # val
            self.validate()
            # testing
            self.test()
            # early stopping on the basis of validation result
            if self.pre_val_metric >= self.cur_val_metric >= 0. or self.finished:
                # update flag
                self.finished = True
                # save log
                end_time = datetime.now()
                self.val_log.append('\nEnd Time: {}'.format(end_time))
                self.val_log.append('\nTotal Time: {}'.format(end_time-self.start_time))
                save_txt(self.config.LOG_POINT.format('val'), self.val_log)
                self.test_log += self.val_log[-2:]
                save_txt(self.config.LOG_POINT.format('test'), self.test_log)
                # save val result
                val_result = ['Src: {}\nTgt: {}\nPred: {}\n\n'.format(
                    x, y, y_) for x, y, y_ in zip(self.val_src, self.val_tgt, self.val_pred)]
                save_txt(self.config.RESULT_POINT.format('val'), val_result)
                # save test result
                test_result = ['Src: {}\nTgt: {}\nPred: {}\n\n'.format(
                    x, y, y_) for x, y, y_ in zip(self.test_src, self.test_tgt, self.test_pred)]
                save_txt(self.config.RESULT_POINT.format('test'), test_result)

            self.epoch += 1

    def validate(self):
        print('\nValidating...')
        # online validate
        model = pick_model(self.config, 'recursion')
        model.load_state_dict(self.model.state_dict())
        all_xs, all_ys, all_ys_ = [], [], []
        valset_generator = tqdm(self.valset_generator)
        model.eval()
        with torch.no_grad():
            for data in valset_generator:
                if self.config.model_name == 'transformer':
                    xs, x_lens, ys, src_mask, tgt_mask = data
                    print(x_lens.cpu().detach().numpy()[0])
                    print(translate(xs.cpu().detach().numpy()[0], self.src_idx2vocab_dict))
                    print(translate(ys.cpu().detach().numpy()[0], self.src_idx2vocab_dict))
                    # break
                    start_symbols = torch.ones([ys.size(0), 1], 
                        dtype=torch.double, device=self.config.device)*self.config.start_idx
                    ys = torch.cat((start_symbols.long(), ys), 1)
                    ys_, _, _ = recursive_infer_transformer(xs, x_lens, ys, model, src_mask, tgt_mask,
                                                            self.config.max_infer_step, self.src_idx2vocab_dict,
                                                            self.src_vocab2idx_dict, self.tgt_idx2vocab_dict, self.config)
                    # break
                else:
                    xs, x_lens, ys = data
                    # print(x_lens.cpu().detach().numpy()[0])
                    # print(translate(xs.cpu().detach().numpy()[0], self.src_idx2vocab_dict))
                    # print(translate(ys.cpu().detach().numpy()[0], self.src_idx2vocab_dict))
                    # break
                    ys_,  _, _ = recursive_infer(xs, x_lens, model, self.config.max_infer_step, 
                        self.src_idx2vocab_dict, self.src_vocab2idx_dict, self.tgt_idx2vocab_dict, self.config)

                xs = xs.cpu().detach().numpy() # batch_size, max_xs_seq_len
                ys = ys.cpu().detach().numpy() # batch_size, max_ys_seq_len
                ys_ = ys_.cpu().detach().numpy() # batch_size, max_ys_seq_len

                # print(translate(xs[0], self.src_idx2vocab_dict))
                # print(translate(ys[0], self.src_idx2vocab_dict))
                # print(translate(ys_[0], self.src_idx2vocab_dict))

                xs, ys, ys_ = rm_pads(xs, ys, ys_, self.config.pad_idx)
                all_xs += xs
                all_ys += ys 
                all_ys_ += ys_

        # evaluation
        eva_matrix = Evaluate(self.config, all_ys, all_ys_, self.src_idx2vocab_dict)
        eva_msg = 'Val Epoch {} Total Step {} '.format(self.epoch, self.step)
        eva_msg += eva_matrix.eva_msg
        print(eva_msg)
        self.val_log.append(eva_msg)
        # random sample to show
        src, tar, pred = rand_sample(all_xs, all_ys, all_ys_, 
            self.src_idx2vocab_dict, self.src_idx2vocab_dict, self.src_idx2vocab_dict)
        print(' src: {}\n tar: {}\n pred: {}'.format(src, tar, pred))
        # early stopping
        if eva_matrix.key_metric >= self.val_key_metric:
            self.val_key_metric = eva_matrix.key_metric
            # save model
            save_check_point(self.step, self.epoch, self.model.state_dict, self.opt.state_dict, self.config.SAVE_POINT)
        self.val_metric_list.append(eva_matrix.key_metric)
        self.pre_val_metric = get_list_mean(self.val_metric_list[-self.config.val_win_size-1: -1])
        self.cur_val_metric = get_list_mean(self.val_metric_list[-self.config.val_win_size:])
        # save test output
        self.val_src = [' '.join(translate(x, self.src_idx2vocab_dict)) for x in all_xs]
        self.val_tgt = [' '.join(translate(y, self.src_idx2vocab_dict)) for y in all_ys]
        self.val_pred = [' '.join(translate(y_, self.src_idx2vocab_dict)) for y_ in all_ys_]

    def test(self):
        print('\nTesting...')
        model = pick_model(self.config, 'recursion')
        # local test
        checkpoint_to_load =  torch.load(self.config.LOAD_POINT, map_location=self.config.device) 
        print('Model restored from {}.'.format(self.config.LOAD_POINT))
        model.load_state_dict(checkpoint_to_load['model'] ) 
        # online test
        # model.load_state_dict(self.model.state_dict())
        all_xs, all_ys, all_ys_ = [], [], []
        testset_generator = tqdm(self.testset_generator)
        model.eval()
        with torch.no_grad():
            for data in testset_generator: 
                if self.config.model_name == 'transformer':
                    xs, x_lens, ys, src_mask, tgt_mask = data
                    # print(x_lens.cpu().detach().numpy()[0])
                    # print(translate(xs.cpu().detach().numpy()[0], self.src_idx2vocab_dict))
                    # print(translate(ys.cpu().detach().numpy()[0], self.src_idx2vocab_dict))
                    # break
                    start_symbols = torch.ones([ys.size(0), 1], dtype=torch.double,
                                               device=self.config.device) * self.config.start_idx
                    ys = torch.cat((start_symbols.long(), ys), 1)
                    ys_, _, _ = recursive_infer_transformer(xs, x_lens, ys, model, src_mask, tgt_mask,
                                                            self.config.max_infer_step,
                                                            self.src_idx2vocab_dict,
                                                            self.src_vocab2idx_dict, self.tgt_idx2vocab_dict,
                                                            self.config)
                else:
                    xs, x_lens, ys = data
                # print(x_lens.cpu().detach().numpy()[0])
                # print(translate(xs.cpu().detach().numpy()[0], self.src_idx2vocab_dict))
                # print(translate(ys.cpu().detach().numpy()[0], self.src_idx2vocab_dict))
                # break
                    ys_, _, _ = recursive_infer(xs, x_lens, model, self.config.max_infer_step, 
                        self.src_idx2vocab_dict, self.src_vocab2idx_dict, self.tgt_idx2vocab_dict, self.config)

                xs = xs.cpu().detach().numpy() # batch_size, max_xs_seq_len
                ys = ys.cpu().detach().numpy() # batch_size, max_ys_seq_len
                ys_ = ys_.cpu().detach().numpy() # batch_size, max_ys_seq_len
                xs, ys, ys_ = rm_pads(xs, ys, ys_, self.config.pad_idx)
                all_xs += xs
                all_ys += ys 
                all_ys_ += ys_

        eva_matrix = Evaluate(self.config, all_ys, all_ys_, self.src_idx2vocab_dict)
        eva_msg = 'Test Epoch {} Total Step {} '.format(self.epoch, self.step)
        eva_msg += eva_matrix.eva_msg
        print(eva_msg)
        self.test_log.append(eva_msg)
        # random sample to show
        src, tar, pred = rand_sample(xs, ys, ys_, 
            self.src_idx2vocab_dict, self.src_idx2vocab_dict, self.src_idx2vocab_dict)
        print(' src: {}\n tar: {}\n pred: {}'.format(src, tar, pred))

        # vocab to index
        # test_all_xs = [translate(x, self.src_idx2vocab_dict) for x in all_xs]
        # test_all_ys = [translate(y, self.src_idx2vocab_dict) for y in all_ys]
        # test_all_ys_ = [translate(y, self.tgt_idx2vocab_dict) for y in all_ys_]

        # for i in range(20):
        #     print('x:', test_all_xs[i])
        #     print('y:', test_all_ys[i])
        #     print('y_', test_all_ys_[i])
        #     print()

        # save test output
        self.test_src = [' '.join(translate(x, self.src_idx2vocab_dict)) for x in all_xs]
        self.test_tgt = [' '.join(translate(y, self.src_idx2vocab_dict)) for y in all_ys]
        self.test_pred = [' '.join(translate(y_, self.src_idx2vocab_dict)) for y_ in all_ys_]

def main(): 
    # initial everything
    te = TextEditor(RecursionConfig())
    # train!
    te.train()

if __name__ == '__main__':
      main()