# Recursive-Text-Editing: Data Preprocessing


## Introduction
This folder contains the code to preprocess data for modeling.


## Parameters
+ vocabulary size  
The number of positive real digits to be involve in the data generation.
+ sequence length  
The length of input sequences.
+ data size  
The size of the generated dataset, which should be smaller than the space size.


## Directory
+ **end2end.py** - to preprocess data for end2end models
+ **recursion.py** - to preprocess data for recursion models
+ **raw** - the raw data
+ **end2end** - the data ready for end2end models
+ **recursion** - the data ready for recursion models
```
data/
├── README.md
├── end2end
├── end2end.ipynb
├── end2end.py
├── raw
├── recursion
├── recursion.ipynb
├── recursion.py
└── utils.py
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
To prepare data for end2end models with:
+ vocabulary size - 10
+ sequence length - 5
+ data size - 10000
```
$ python end2end.py --num_size 10 --seq_len 5 --data_size 10000
```
To prepare data for recursion models with the same settings:
```
$ python recursion.py --num_size 10 --seq_len 5 --data_size 10000
```


## Output
```
train sample size 7000
train label size 7000
val sample size 1500
val label size 1500
test sample size 1500
test label size 1500
```


## Authors
* **Ning Shi** - mrshininnnnn@gmail.com