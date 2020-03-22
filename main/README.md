# Learnable Grading Formal Language Parser

## Introduction
The goal of this project is to parse questions in natural language to their formal language version according to a certain rule.

## Directory
+ **train.py** - to start the training process
+ **config.py** - to save all the configuration
+ **res/data/prep_data.ipynb** - to format data for the model
+ **res/log** - to save check points
```
main/
├── README.md
├── config.py
├── requirements.txt
├── res
│   ├── data
│   │   ├── gen
│   │   │   ├── ch
│   │   │   ├── mixed
│   │   │   └── non_ch
│   │   ├── prep_data.ipynb
│   │   └── real
│   └── log
├── src
│   ├── models
│   │   └── seq2seq_bi_lstm_att.py
│   └── utils
│       ├── eva.py
│       └── load.py
└── train.py
```

## Dependencies
+ python >= 3.7.6
+ jupyter >= 1.0.0
+ numpy >= 1.18.1
+ torch >= 1.4.0
+ tqdm >= 4.42.1


## Setup
Please ensure the following packages are already installed. A virtual environment is recommended.
+ Python (for .py)
+ Jupyter Notebook (for .ipynb)

```
$ cd main
$ pip3 install pip --upgrade
$ pip3 install -r requirements.txt
$ pip3 install torch torchvision --upgrade
```

## Run
Before running anything, please double check the dataset in **prep_data.ipynb** first. A wrong type of data will cause a failed running. For the first time, please look into ***config.py*** to verify all the configuration settings including file paths, training strategy, and model hyper-parameters. The training should be on an assigned server in a tmux session so that there won't be any unexpected stop. Before any running, please backup existing check points first.
```
$ jupyter notebook # double check the dataset first
$ vim config.py # double check the configuration
$ python3 train.py # start the training
```

## Output
If everything goes well, you may see a similar progressing shown as below.
```
*General Setting*
model: seq2seq_bi_lstm_att
trainable parameters:9,602,828
model's state_dict:
encoder.embedding.weight       torch.Size([1292, 512])
encoder.lstm.weight_ih_l0      torch.Size([1024, 512])
encoder.lstm.weight_hh_l0      torch.Size([1024, 256])
encoder.lstm.bias_ih_l0        torch.Size([1024])
encoder.lstm.bias_hh_l0        torch.Size([1024])
encoder.lstm.weight_ih_l0_reverse    torch.Size([1024, 512])
encoder.lstm.weight_hh_l0_reverse    torch.Size([1024, 256])
encoder.lstm.bias_ih_l0_reverse      torch.Size([1024])
encoder.lstm.bias_hh_l0_reverse      torch.Size([1024])
encoder.lstm.weight_ih_l1      torch.Size([1024, 512])
encoder.lstm.weight_hh_l1      torch.Size([1024, 256])
encoder.lstm.bias_ih_l1        torch.Size([1024])
encoder.lstm.bias_hh_l1        torch.Size([1024])
encoder.lstm.weight_ih_l1_reverse    torch.Size([1024, 512])
encoder.lstm.weight_hh_l1_reverse    torch.Size([1024, 256])
encoder.lstm.bias_ih_l1_reverse      torch.Size([1024])
encoder.lstm.bias_hh_l1_reverse      torch.Size([1024])
encoder.h_out.weight     torch.Size([512, 1024])
encoder.h_out.bias       torch.Size([512])
encoder.c_out.weight     torch.Size([512, 1024])
encoder.c_out.bias       torch.Size([512])
decoder.embedding.weight       torch.Size([1292, 512])
decoder.attn.v     torch.Size([512])
decoder.attn.attn.weight       torch.Size([512, 1536])
decoder.attn.attn.bias   torch.Size([512])
decoder.lstm.weight_ih_l0      torch.Size([2048, 512])
decoder.lstm.weight_hh_l0      torch.Size([2048, 512])
decoder.lstm.bias_ih_l0        torch.Size([2048])
decoder.lstm.bias_hh_l0        torch.Size([2048])
decoder.attn_combine.weight    torch.Size([512, 1024])
decoder.attn_combine.bias      torch.Size([512])
decoder.out.weight       torch.Size([1292, 512])
decoder.out.bias   torch.Size([1292])
device: cpu
use gpu: False
device num: 0
optimizer: Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.0001
    weight_decay: 0
)
train size: 55752
valid size: 2000
test size: 2000
vocab size: 1292
batch size: 128
train batch: 435
valid batch: 16
test batch: 16

reuse model: False


Training...
  0%|                                             | 0/435[00:13<?, ?it/s]
```

## Note
+ Please double check the dataset in jupyter notebook first.
+ Please try to totally understand the ***config.py***.
+ To track the training progress and avoid any unexpected issue, the model is kept very often, which may cause the occupation of large disk space. Please remember to move previous checkpoints to other places for free space. 


## Authors
* **Ning Shi** - ning.shi@learnable.ai
