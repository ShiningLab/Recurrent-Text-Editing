# Recursive-Text-Editing: Main Pipeline


## Introduction


## Directory
```
main/
├── README.md
├── config.py
├── res
├── src
├── train_end2end.py
└── train_recursion.py
```


## Dependencies
+


## Setup
Please ensure the following packages are already installed. A virtual environment is recommended.
+ Python (for .py)
+ Jupyter Notebook (for .ipynb)

```
$ cd main
$ pip3 install pip --upgrade
$ pip3 install -r requirements.txt
$ pip3 install torch --upgrade
```

## Run
```
$ vim config.py
$ python3 train_end2end.py
```


## Output
If everything goes well, you may see a similar progressing shown as below.
```
*Configuration*
model: bi_lstm_rnn_att
trainable parameters:5,017,618
model's state_dict:
encoder.embedding.weight     torch.Size([17, 512])
encoder.lstm.weight_ih_l0    torch.Size([1024, 512])
encoder.lstm.weight_hh_l0    torch.Size([1024, 256])
encoder.lstm.bias_ih_l0      torch.Size([1024])
encoder.lstm.bias_hh_l0      torch.Size([1024])
encoder.lstm.weight_ih_l0_reverse    torch.Size([1024, 512])
encoder.lstm.weight_hh_l0_reverse    torch.Size([1024, 256])
encoder.lstm.bias_ih_l0_reverse      torch.Size([1024])
encoder.lstm.bias_hh_l0_reverse      torch.Size([1024])
decoder.embedding.weight     torch.Size([18, 512])
decoder.attn.v   torch.Size([512])
decoder.attn.attn.weight     torch.Size([512, 1536])
decoder.attn.attn.bias   torch.Size([512])
decoder.attn_combine.weight      torch.Size([512, 1024])
decoder.attn_combine.bias    torch.Size([512])
decoder.lstm.weight_ih_l0    torch.Size([2048, 512])
decoder.lstm.weight_hh_l0    torch.Size([2048, 512])
decoder.lstm.bias_ih_l0      torch.Size([2048])
decoder.lstm.bias_hh_l0      torch.Size([2048])
decoder.out.weight   torch.Size([18, 512])
decoder.out.bias     torch.Size([18])
device: cpu
use gpu: False
train size: 7000
val size: 1500
test size: 1500
source vocab size: 17
target vocab size: 18
batch size: 512
train batch: 13
val batch: 3
test batch: 3

if load check point: False

Training...
  0%|                                                                                                                                                         | 0/13 [00:00<?, ?it/s]
```


## Note


## Authors
* **Ning Shi** - ning.shi@learnable.ai