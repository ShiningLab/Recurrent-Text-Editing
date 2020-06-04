# Recurrent-Text-Editing: Main Pipeline

## Introduction
This folder contains code to reproduce the experiments in the paper *Recurrent Inference in Text Editing*.

## Directory
+ **res/data** for data pre-processing
+ **res/log** for training logs
+ **res/checkpoints** for saving model checkpoints
+ **res/result** for final output on test sets
+ **train_e2e.py** for training End2end models
+ **train_tag.py** for training Tagging models
+ **train_rec.py** for training Recurrence models
+ **config.py** for training configurations
```
main/
├── README.md
├── config.py
├── requirements.txt
├── res
│   ├── check_points
│   ├── data
│   │   ├── aec
│   │   ├── aes
│   │   └── aor
│   ├── log
│   └── result
├── src
│   ├── models
│   └── utils
├── train_e2e.py
├── train_rec.py
└── train_tag.py
```

## Dependencies
+ Python >= 3.7.7 
+ tqdm >= 4.46.0 
+ numpy >= 1.18.4 
+ torch >= 1.5.0
+ python_Levenshtein >= 0.12.0 


## Setup
Please ensure required packages are already installed. A virtual environment is recommended.
```
$ pip install pip --upgrade
$ pip install -r requirements.txt
$ pip install torch --upgrade
```

## Process
1. Generate raw datasets under **code/data/**
2. Copy raw datasets from **code/data/** to **code/main/res/data/**
3. Pre-process datasets under **code/main/res/data/**
4. Start training under **code/main/**

## Configuration
+ data_src - AOR, AES, or AEC
+ method - End2end, Tagging, or Recurrence
+ data_mode - Offline Training or Online Training
+ model_name
    + gru_rnn
    + lstm_rnn
    + bi_gru_rnn
    + bi_lstm_rnn
    + bi_gru_rnn_att
    + bi_lstm_rnn_att
    + transformer
+ N - the number of unique integers
+ L - the number of integers in an equation
+ D - the number of unique equations

## Run
Here is an example command to train an End2end model. Please take a look at the **config.py** to ensure training configurations and then run the training script **train_e2e.py**.
```
$ vim config.py
$ python train_e2e.py
```

## Output
If everything goes well, you should see a similar progressing shown as below.
```
*Configuration*
model: bi_lstm_rnn_att
trainable parameters:5,017,106
model state_dict:
encoder.embedding.weight    torch.Size([16, 512])
encoder.bi_lstm.weight_ih_l0    torch.Size([1024, 512])
encoder.bi_lstm.weight_hh_l0    torch.Size([1024, 256])
encoder.bi_lstm.bias_ih_l0  torch.Size([1024])
encoder.bi_lstm.bias_hh_l0  torch.Size([1024])
encoder.bi_lstm.weight_ih_l0_reverse    torch.Size([1024, 512])
encoder.bi_lstm.weight_hh_l0_reverse    torch.Size([1024, 256])
encoder.bi_lstm.bias_ih_l0_reverse  torch.Size([1024])
encoder.bi_lstm.bias_hh_l0_reverse  torch.Size([1024])
decoder.embedding.weight    torch.Size([18, 512])
decoder.attn.v  torch.Size([512])
decoder.attn.attn.weight    torch.Size([512, 1536])
decoder.attn.attn.bias  torch.Size([512])
decoder.attn_combine.weight torch.Size([512, 1024])
decoder.attn_combine.bias   torch.Size([512])
decoder.lstm.weight_ih_l0   torch.Size([2048, 512])
decoder.lstm.weight_hh_l0   torch.Size([2048, 512])
decoder.lstm.bias_ih_l0 torch.Size([2048])
decoder.lstm.bias_hh_l0 torch.Size([2048])
decoder.out.weight  torch.Size([18, 512])
decoder.out.bias    torch.Size([18])
device: cpu
use gpu: False
train size: 7000
val size: 1500
test size: 1500
source vocab size: 16
target vocab size: 18
batch size: 256
train batch: 27
val batch: 6
test batch: 6
if load check point: False


Training...
  4%|██▍                                                             | 1/27 [00:01<00:38,  1.48s/it]
```

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com