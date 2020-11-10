# Recurrent-Text-Editing

This repository is for the paper [Recurrent Inference in Text Editing](https://www.aclweb.org/anthology/2020.findings-emnlp.159/) in [Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings](https://www.aclweb.org/anthology/volumes/2020.findings-emnlp/).

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

## BibTex
```
@inproceedings{shi-etal-2020-recurrent,
    title = "Recurrent Inference in Text Editing",
    author = "Shi, Ning  and
      Zeng, Ziheng  and
      Zhang, Haotian  and
      Gong, Yichen",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.159",
    pages = "1758--1769",
    abstract = "In neural text editing, prevalent sequence-to-sequence based approaches directly map the unedited text either to the edited text or the editing operations, in which the performance is degraded by the limited source text encoding and long, varying decoding steps. To address this problem, we propose a new inference method, Recurrence, that iteratively performs editing actions, significantly narrowing the problem space. In each iteration, encoding the partially edited text, Recurrence decodes the latent representation, generates an action of short, fixed-length, and applies the action to complete a single edit. For a comprehensive comparison, we introduce three types of text editing tasks: Arithmetic Operators Restoration (AOR), Arithmetic Equation Simplification (AES), Arithmetic Equation Correction (AEC). Extensive experiments on these tasks with varying difficulties demonstrate that Recurrence achieves improvements over conventional inference methods.",
}
```
