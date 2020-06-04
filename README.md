# Recurrent-Text-Editing

This repository is for the paper *Recurrent Inference in Text Editing*.

## Methods
+ End2end
+ Tagging
+ Recurrence

## Models
+ Naive GRU RNN
+ Naive LSTM RNN
+ Bi-directional GRU RNN
+ Bi-directional LSTM RNN
+ Bi-directional GRU RNN with Attention
+ Bi-directional LSTM RNN with Attention
+ Transformer

## Data
+ Arithmetic Operators Restoration (AOR)
+ Arithmetic Equation Simplification (AES)
+ Arithmetic Equation Correction (AEC)

## Directory
+ **data** - for data generation
+ **main** - for training 
+ **exp_log** - validation and testing performance during training for experiments in the original work
```
code/
├── README.md
├── data
├── main
├── exp_log
└── reference
```

## Process
1. Generate raw datasets under **code/data/**
2. Copy raw datasets from **code/data/** to **code/main/res/data/**
3. Pre-process datasets under **code/main/res/data/**
4. Start training under **code/main/**

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com
* **Ziheng Zeng**