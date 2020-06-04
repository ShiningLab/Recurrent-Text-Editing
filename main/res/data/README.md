# Recurrent-Text-Editing: Data Pre-processing

## Introduction
This folder contains code to pre-process raw datasets.

## Parameters
+ N - the number of unique integers
+ L - the number of integers in an equation
+ D - the number of unique equations

## Directory
+ **aor** - For AEC raw datasets pre-processing
+ **aes** - For AEC raw datasets pre-processing
+ **aec** - For AEC raw datasets pre-processing
```
data/
├── README.md
├── requirements.txt
├── aec
├── aes
└── aor
```

## Dependencies
+ python >= 3.7.7
+ numpy >= 1.18.4 
+ python_Levenshtein >= 0.12.0

## Run
To prepare data for End2end on AOR with:
+ N - 10
+ L - 5
+ D - 10000
```
$ cd aor
$ python e2e.py --N 10 --L 5 --D 10000
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