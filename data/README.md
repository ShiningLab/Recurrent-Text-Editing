# Recursive-Text-Editing: Data

## Introduction
This folder contains the code to generate the raw dataset for the Arithmetic Operators Insertion (AOI) domain. To solve the task, given a sequence of positive real numbers, a good learner should insert appropriate mathematical operators between two numbers to hold a valid equation. For example, the model should insert "+" between "1" and "1" and "==" between "1" and "2" for the sequence of positive real numbers "1 1 2" to hold a valid equation "1 + 1 == 2." The penultimate is always the equation mark "==" and the last one is always the value of the left side.

## Parameters
+ vocabulary size  
The number of positive real digits to be involve in the data generation.
+ sequence length  
The length of input sequences.
+ data size  
The size of the generated dataset, which should be smaller than the space size.

## Directory
+ **generator.py** - to generate the dataset given several parameters
+ **generator.ipynb** - to go through the process step by step
+ **x.txt** - the input sequences
+ **y.txt** - the output sequences
```
data/
├── README.md
├── generator.ipynb
├── generator.py
├── raw
│   └── vocab_size_10
│       ├── seq_len_1
│       │   └── data_size_10
│       │       ├── x.txt
│       │       └── y.txt
│       ├── seq_len_2
│       ├── seq_len_3
│   └── vocab_size_11
│   └── vocab_size_12
└── utilities
```

## Dependencies
+ python >= 3.7.6
+ jupyter >= 1.0.0
+ numpy >= 1.16.2

## Setup
Please ensure the following packages are already installed. A virtual environment is recommended.
+ Python (for .py)
+ Jupyter Notebook (for .ipynb)

```
$ cd main
$ pip3 install pip --upgrade
$ pip3 install -r requirements.txt
```

## Run
To generate a data with:
+ vocabulary size - 10
+ sequence length - 5
+ data size - 10000
```
$ python generator.py --vocab_size 10 --seq_len 5 --data_size 10000
```

## Output
```
100%|█████████████████████████████████| 10000/10000 [00:12<00:00, 783.65it/s]
train size 7000 (7000, 2)
val size 1500 (1500, 2)
test size 1500 (1500, 2)
find output from raw/vocab_size_10/seq_len_5/data_size_10000
```


## Examples
+ 5 9 9 4 10 -> 5 + 9 / 9 + 4 == 10
+ 10 2 5 2 2 -> 10 / 2 / 5 * 2 == 2
+ 3 4 8 4 8 -> 3 * 4 - 8 + 4 == 8
+ etc.


## Note
+ The operators used to yield a valid equation are not unique, thus we can evaluate the performance by examining the equation in addition to token accuracy and sequence accuracy.
+ Using "==" instead of "=" as the equation mark is for the convenience to evaluate the equation by Python built-in function eval().
+ There are a total of three sub sets for a generation, namely, train set, val set, and test set.
+ Number "0" and "1" are not involved to avoid one-to-many cases.


## Authors
* **Ning Shi** - mrshininnnnn@gmail.com
