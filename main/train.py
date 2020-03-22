#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'

import torch
from torch.utils import data as torch_data

import os
import random
from tqdm import tqdm

from torch_seq2seq.config import Config
from torch_seq2seq.src.utils.load import load_json
from torch_seq2seq.src.utils.eva import Evaluate as eva
from torch_seq2seq.src.models import seq2seq_bi_lstm_att

from data_generation.data_generator_api import generate_mixed

class OnlineDataset(torch_data.Dataset):
      """Custom data.Dataset compatible with data.DataLoader."""
      def __init__(self, x, y): 
            super(OnlineDataset, self).__init__()
            self.src, self.tgt = x, y
            self.data_size = len(x)

      def __len__(self):
            return self.data_size

      def __getitem__(self, idx):
            return self.src[idx], self.tgt[idx]

class Dataset(torch_data.Dataset):
      """Custom data.Dataset compatible with data.DataLoader."""
      def __init__(self, data_dict):
            super(Dataset, self).__init__()
            self.src = data_dict['src']
            self.tgt = data_dict['tgt']
            self.data_size = len(self.src)

      def __len__(self):
            return self.data_size

      def __getitem__(self, idx):
            return self.src[idx], self.tgt[idx]

class GradingParser():
      """docstring for GradingParser"""
      def __init__(self):
            super(GradingParser).__init__()
            self.config = Config()
            self.setup_gpu()
            self.load_vocab()
            self.update_vocab()
            # self.load_data()
            # self.update_config()
            # self.prepare_train()
            # self.setup_model()

      def setup_gpu(self):
            # confirm the device which can be either cpu or gpu
            self.config.use_gpu = torch.cuda.is_available()
            self.num_device = torch.cuda.device_count()
            if self.config.use_gpu:
                  self.config.device = 'cuda'
                  torch.backends.cudnn.benchmark = True
                  torch.backends.cudnn.deterministic = True
                  if self.num_device <= 1:
                        self.config.data_parallel = False
                  elif self.config.data_parallel:
                              torch.multiprocessing.set_start_method('spawn', force=True)
            else:
                  self.config.device = 'cpu'
                  self.config.data_parallel = False

      def load_vocab(self):
            # load the vocab dictionary and update config
            self.vocab2idx_dict = load_json(self.config.VOCAB_PATH)
            print(self.vocab2idx_dict)
            # self.idx2vocab_dict = {v: k for k, v in self.vocab2idx_dict.items()}
            # self.config.pad_idx = self.vocab2idx_dict[self.config.pad_symbol]
            # self.config.start_idx = self.vocab2idx_dict[self.config.start_symbol]
            # self.config.end_idx = self.vocab2idx_dict[self.config.end_symbol]

      def update_vocab(self):
            # TODO: remove after having uniform vocab dict
            # update current vocab dict
            i = len(self.vocab2idx_dict)
            for token in self.train_vocab:
                  if token not in self.vocab2idx_dict:
                        self.vocab2idx_dict[token] = i 
                        i += 1
            print(self.vocab2idx_dict)
            self.idx2vocab_dict = {v: k for k, v in self.vocab2idx_dict.items()}

      def online_collate_fn(self, data):
            # a customized collate function used in the data loader
            def preprocess(xs, ys):
                  # split new line symbol \n
                  xs = [x.strip('\n') for x in xs]
                  ys = [y.strip('\n') for y in ys]
                  # white space tokenization
                  xs = [x.split() for x in xs]
                  ys = [y.split() for y in ys]
                  # vocab to index
                  xs = [self.vocab_to_index(x) for x in xs]
                  ys = [self.vocab_to_index(y) for y in ys]
                  # add start and end symbol
                  xs = [torch.Tensor([self.config.start_idx] + x + [self.config.end_idx]) for x in xs]
                  ys_in = [torch.Tensor([self.config.start_idx] + y) for y in ys]
                  ys_out = [torch.Tensor(y + [self.config.end_idx]) for y in ys]
                  return xs, ys_in, ys_out

            def padding(seqs):
                  seq_length_list = [len(seq) for seq in seqs]
                  padded_seqs = torch.zeros([len(seqs), max(seq_length_list)], dtype=torch.int64)
                  for i, seq in enumerate(seqs):
                        seq_length = seq_length_list[i]
                        padded_seqs[i, :seq_length] = seq[:seq_length]
                  return padded_seqs, seq_length_list

            data.sort(key=len, reverse=True)
            xs, ys = zip(*data)
            xs, ys_in, ys_out = preprocess(xs, ys)
            xs, x_lens = padding(xs)
            ys_in, y_lens = padding(ys_in)
            ys_out, _ = padding(ys_out)

            return (xs.to(self.config.device), 
                  torch.Tensor(x_lens).to(self.config.device), 
                  ys_in.to(self.config.device), 
                  ys_out.to(self.config.device), 
                  torch.Tensor(y_lens).to(self.config.device))


      def collate_fn(self, data):
            # a customized collate function used in the data loader
            def preprocess(xs, ys):
                  # add start and end symbol
                  xs = [torch.Tensor([self.config.start_idx] + x + [self.config.end_idx]) for x in xs]
                  ys_in = [torch.Tensor([self.config.start_idx] + y) for y in ys]
                  ys_out = [torch.Tensor(y + [self.config.end_idx]) for y in ys]
                  return xs, ys_in, ys_out

            def padding(seqs):
                  seq_length_list = [len(seq) for seq in seqs]
                  padded_seqs = torch.zeros([len(seqs), max(seq_length_list)], dtype=torch.int64)
                  for i, seq in enumerate(seqs):
                        seq_length = seq_length_list[i]
                        padded_seqs[i, :seq_length] = seq[:seq_length]
                  return padded_seqs, seq_length_list

            data.sort(key=len, reverse=True)
            xs, ys = zip(*data)
            xs, ys_in, ys_out = preprocess(xs, ys)
            xs, x_lens = padding(xs)
            ys_in, y_lens = padding(ys_in)
            ys_out, _ = padding(ys_out)

            return (xs.to(self.config.device), 
                  torch.Tensor(x_lens).to(self.config.device), 
                  ys_in.to(self.config.device), 
                  ys_out.to(self.config.device), 
                  torch.Tensor(y_lens).to(self.config.device))

      def load_data(self):
            # read data dictionary from json file
            self.data_dict = load_json(self.config.DATA_PATH)
            # generate online training data set
            self.train_x, self.train_y, self.train_vocab = generate_mixed(
                  iterations=[self.config.sub_data_size, 
                  self.config.sub_data_size, 
                  self.config.train_data_size - 2 * self.config.sub_data_size], 
                  replace_unk=True, unk_serialized=False)
            # train data loader
            # self.train_dataset = Dataset(self.data_dict['train'])
            self.train_dataset = OnlineDataset(self.train_x, self.train_y)
            self.trainset_generator = torch_data.DataLoader(
                  self.train_dataset, 
                  batch_size=self.config.batch_size, 
                  collate_fn=self.online_collate_fn, 
                  shuffle=self.config.shuffle, 
                  drop_last=self.config.drop_last)
            # valid data loader
            self.valid_dataset = Dataset(self.data_dict['valid'])
            self.validset_generator = torch_data.DataLoader(
                  self.valid_dataset, 
                  batch_size=self.config.batch_size, 
                  collate_fn=self.collate_fn, 
                  shuffle=False, 
                  drop_last=False)
            # test data loader
            self.test_dataset = Dataset(self.data_dict['test'])
            self.testset_generator = torch_data.DataLoader(
                  self.test_dataset, 
                  batch_size=self.config.batch_size, 
                  collate_fn=self.collate_fn, 
                  shuffle=False,
                  drop_last=False)

      def update_config(self):
            # updated config according to data and vocab
            def get_batch_size(dataset_size):
                  if dataset_size % self.config.batch_size == 0:
                        return dataset_size // self.config.batch_size
                  else:
                        return dataset_size // self.config.batch_size + 1

            self.config.vocab_size = len(self.vocab2idx_dict)
            self.config.train_size = len(self.train_dataset)
            self.config.train_batch = len(self.trainset_generator)
            # self.config.train_batch = get_batch_size(self.config.train_size)
            self.config.valid_size = len(self.valid_dataset)
            self.config.valid_batch = len(self.validset_generator)
            # self.config.valid_batch = get_batch_size(self.config.valid_size)
            self.config.test_size = len(self.test_dataset)
            self.config.test_batch = len(self.testset_generator)
            # self.config.test_batch = get_batch_size(self.config.test_size)

      def prepare_train(self):
            # preparation for training
            self.step = 0
            self.epoch = 0
            self.finished = False
            self.valid_epoch, self.valid_step,  = 0, 0 
            self.valid_loss, self.valid_token_acc, self.valid_seq_acc = float('-inf'), float('-inf'), float('-inf')

      def pick_model(self):
            # for switching model
            if self.config.model_name == 'seq2seq_bi_lstm_att':
                  return seq2seq_bi_lstm_att.ModelGraph(self.config).to(self.config.device)

      def prepare_optimizer(self):
            self.opt = torch.optim.Adam(
                  self.model.parameters(), 
                  lr=self.config.learning_rate)

      def load_check_point(self):
            checkpoint_to_load =  torch.load(self.config.LOAD_POINT, map_location=self.config.device)
            self.step = checkpoint_to_load['step']
            self.epoch = checkpoint_to_load['epoch']
            model_state_dict = checkpoint_to_load['model']
            self.model.load_state_dict(model_state_dict)
            self.opt.load_state_dict(checkpoint_to_load['optimizer'])

      def save_check_point(self):
            # save model, optimizer, and everything required to keep
            checkpoint_to_save = {
            'step': self.step, 
            'epoch': self.epoch, 
            'model': self.model.state_dict(), 
            'optimizer': self.opt.state_dict()}
            save_path = self.config.SAVE_POINT.format(self.step, self.epoch)
            torch.save(checkpoint_to_save, save_path)
            print('Model saved as {}.'.format(save_path))

      def setup_model(self):
            # initialize model weights, optimizer, and loss function
            self.model = self.pick_model()
            def init_weights(model):
                  for name, param in model.named_parameters():
                        if 'weight' in name:
                              torch.nn.init.normal_(param.data, mean=0, std=0.01)
                        else:
                              torch.nn.init.constant_(param.data, 0)
            self.model.apply(init_weights)
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_idx)
            self.prepare_optimizer()
            if self.config.reuse_model:
                  self.load_check_point()
            if self.config.data_parallel:
                  self.model = torch.nn.DataParallel(self.model)

      def count_parameters(self):
            # get total size of trainable parameters
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

      def show_info(self):
            # show general information before training
            if self.config.general_show:
                  print('\n*General Setting*')
                  print('model:', self.config.model_name)
                  print('trainable parameters:{:,.0f}'.format(self.count_parameters()))
                  print("model's state_dict:")
                  for param_tensor in self.model.state_dict():
                        print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
                  print('device:', self.config.device)
                  print('use gpu:', self.config.use_gpu)
                  print('device num:', self.num_device)
                  print('optimizer:', self.opt)
                  print('train size:', self.config.train_size)
                  print('valid size:', self.config.valid_size)
                  print('test size:', self.config.test_size)
                  print('vocab size:', self.config.vocab_size)
                  print('batch size:', self.config.batch_size)
                  print('train batch:', self.config.train_batch)
                  print('valid batch:', self.config.valid_batch)
                  print('test batch:', self.config.test_batch)
                  print('\nreuse model:', self.config.reuse_model)
                  if self.config.reuse_model:
                        print('Model restored from {}.'.format(self.config.LOAD_POINT))
                  print()

      def index_to_vocab(self, idx_seq: list) -> list:
            return [self.idx2vocab_dict[idx] for idx in idx_seq]

      def vocab_to_index(self, tk_seq: list) -> list:
            return [self.vocab2idx_dict[token] for token in tk_seq]

      def update_step(self, loss):
            # update process in each train step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clipping_threshold)
            self.opt.step()
            self.opt.zero_grad()
            return loss.item()

      def rm_pad(self, tk_seq):
            return [t for t in tk_seq if t != self.config.pad_idx]

      def prepare_output(self, srcs, tars, preds):
            srcs = [self.rm_pad(seq) for seq in srcs]
            tars = [self.rm_pad(seq) for seq in tars]
            preds = [p_seq[:len(t_seq)] for p_seq, t_seq in zip(preds, tars)]
            return srcs, tars, preds

      def rand_sample(self, srcs, tars, preds):
            src, tar, pred = random.choice([(src, tar, pred) for src, tar, pred in zip(srcs, tars, preds)])
            src = self.index_to_vocab(src)
            tar = self.index_to_vocab(tar)
            pred = self.index_to_vocab(pred)
            return ' '.join(src), ' '.join(tar), ' '.join(pred)

      def train_op(self):
            # training operation
            self.show_info()
            while not self.finished:
                  print('\nTraining...')
                  self.model.train()
                  # generate online training data set for each epoch
                  self.train_x, self.train_y, self.train_vocab = generate_mixed(
                        iterations=[self.config.sub_data_size, 
                              self.config.sub_data_size, 
                              self.config.train_data_size - 2 * self.config.sub_data_size], 
                        replace_unk=True, unk_serialized=False)
                  # train data loader
                  trainset_generator = tqdm(self.trainset_generator)
                  for i, (xs, xs_lens, ys_in, ys_out, ys_lens) in enumerate(trainset_generator): 

                        # print(self.index_to_vocab(xs.numpy()[0]))
                        # print()
                        # print(self.index_to_vocab(ys_in.numpy()[0]))
                        # print()
                        # print(self.index_to_vocab(ys_out.numpy()[0]))
                        # break

                        logits = self.model(xs, xs_lens, ys_in, ys_out, ys_lens, self.config.teacher_forcing_ratio)
                        loss = self.criterion(logits.view(-1, self.config.vocab_size), ys_out.view(-1))
                        loss = self.update_step(loss)

                        srcs = xs.cpu().detach().numpy()
                        tars = ys_out.cpu().detach().numpy()
                        preds = torch.argmax(logits, dim=2).cpu().detach().numpy()

                        if (i / self.config.train_batch) % self.config.progress_rate == 0:
                              # post-processing
                              srcs, tars, preds = self.prepare_output(srcs, tars, preds)
                              # evaluation
                              eva_matrix = eva(self.config, tars, preds)
                              # update processing bar
                              trainset_generator.set_description('Train Epoch {} Total Step {} Loss {:.4f} Token Acc:{:.4f} Seq Acc:{:.4f}'.format(
                                    self.epoch, self.step, loss, eva_matrix.token_acc, eva_matrix.seq_acc))
                              trainset_generator.refresh()
                              # random sample to show
                              src, tar, pred = self.rand_sample(srcs, tars, preds)
                              print(' src: {}\n tar: {}\n pred: {}'.format(src, tar, pred))
                        
                        self.step += 1

                  if self.epoch >= self.config.test_epoch: 
                        self.finished = True

                  self.early_stop()
                  self.infer()
                  self.epoch += 1
                  # TODO: fix the vocab size
                  # can not train the second epoch due to dynamic embedding layer
                  break

                  self.train_dataset = OnlineDataset(self.train_x, self.train_y)
                  # TODO: remove once there is a vocab dict file to load
                  # update current vocab dict 
                  self.update_vocab()
                  self.trainset_generator = torch_data.DataLoader(
                        self.train_dataset, 
                        batch_size=self.config.batch_size, 
                        collate_fn=self.online_collate_fn, 
                        shuffle=self.config.shuffle, 
                        drop_last=self.config.drop_last)

      def early_stop(self):
            print('\nValidating...')
            all_valid_loss = 0
            all_valid_srcs, all_valid_tars, all_valid_preds = [], [], []
            validset_generator = tqdm(self.validset_generator)
            self.model.eval()
            with torch.no_grad():
                  for xs, xs_lens, ys_in, ys_out, ys_lens in validset_generator:
                        logits = self.model(xs, xs_lens, ys_in, ys_out, ys_lens, 0)
                        loss = self.criterion(logits.view(-1, self.config.vocab_size), ys_out.view(-1)).item()

                        srcs = xs.cpu().detach().numpy()
                        tars = ys_out.cpu().detach().numpy()
                        preds = torch.argmax(logits, dim=2).cpu().detach().numpy()
                        srcs, tars, preds = self.prepare_output(srcs, tars, preds)

                        all_valid_loss += loss
                        all_valid_srcs += srcs
                        all_valid_tars += tars
                        all_valid_preds += preds

            all_valid_loss /= self.config.valid_batch
            eva_matrix = eva(self.config, all_valid_tars, all_valid_preds)
            print('Validation Epoch:{} Total Step:{} Loss {:.4f} Token Acc:{:.4f} Seq Acc:{:.4f}'.format(
                  self.epoch, self.step, all_valid_loss, eva_matrix.token_acc, eva_matrix.seq_acc))

            if self.config.early_stop == 'seq_acc' and eva_matrix.seq_acc >= self.valid_seq_acc:
                  self.valid_epoch, self.valid_step = self.epoch, self.step
                  self.valid_loss = all_valid_loss
                  self.valid_token_acc = eva_matrix.token_acc
                  self.valid_seq_acc = eva_matrix.seq_acc
                  self.save_check_point()

            src, tar, pred = self.rand_sample(all_valid_srcs, all_valid_tars, all_valid_preds)
            print(' src: {}\n tar: {}\n pred: {}'.format(src, tar, pred))

            print('Best Validation Epoch:{} Step:{} Loss {:.4f} Token Acc:{:.4f} Seq Acc:{:.4f}'.format(
                  self.valid_epoch, self.valid_step, self.valid_loss, self.valid_token_acc, self.valid_seq_acc))

      def infer(self, test=False):
            all_test_loss = 0
            all_test_srcs, all_test_tars, all_test_preds = [], [], []
            print('\nTesting...')
            model = self.pick_model()
            load_points = [p for p in os.listdir(self.config.SAVE_PATH) if p.endswith('.pt')]
            load_point = sorted(load_points, key=lambda x: int(x.split('_')[0]))[-1]
            load_point = os.path.join(self.config.SAVE_PATH, load_point)
            checkpoint_to_load =  torch.load(load_point, map_location=self.config.device)
            step = checkpoint_to_load['step']
            epoch = checkpoint_to_load['epoch']

            if (step == self.step and epoch == self.epoch) or test:
                  model_state_dict = checkpoint_to_load['model']
                  model.load_state_dict(model_state_dict)
                  print('Model restored from {}.'.format(self.config.LOAD_POINT))

                  testset_generator = tqdm(self.testset_generator)
                  model.eval()
                  with torch.no_grad():
                        for xs, xs_lens, ys_in, ys_out, ys_lens in testset_generator:
                              logits = self.model(xs, xs_lens, ys_in, ys_out, ys_lens, 0)
                              loss = self.criterion(logits.view(-1, self.config.vocab_size), ys_out.view(-1)).item()

                              srcs = xs.cpu().detach().numpy()
                              tars = ys_out.cpu().detach().numpy()
                              preds = torch.argmax(logits, dim=2).cpu().detach().numpy()
                              srcs, tars, preds = self.prepare_output(srcs, tars, preds)

                              all_test_loss += loss
                              all_test_srcs += srcs
                              all_test_tars += tars
                              all_test_preds += preds

                  all_test_loss /= self.config.test_batch
                  eva_matrix = eva(self.config, all_test_tars, all_test_preds)
                  print('Test Epoch:{} Total Step:{} Loss {:.4f} Token Acc:{:.4f} Seq Acc:{:.4f}'.format(
                        epoch, step, all_test_loss, eva_matrix.token_acc, eva_matrix.seq_acc))

def main():
      # initial everything
      gp = GradingParser()
      # training
      # gp.train_op()
      # testing
      # gp.infer(True)

if __name__ == '__main__':
      main()