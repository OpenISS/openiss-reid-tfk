# Person Re-Identification Baseline

This repo basically is the re-implementation (tensorflow + keras) of a person
re-identification baseline propose by the paper
["Bag of Tricks and A Strong Baseline for Deep Person Re-identification"](https://arxiv.org/abs/1903.07071).

The authors original implementation which is in Pytorch can be found in their
[repo](https://github.com/michuanhaohao/reid-strong-baseline).


## Enviornment

A powerful GPU is required for running the code, with Nivida GTX 1070ti, a training with the standard 120 epochs
will take almost 4 hours.

This implementatoin is based on tensorflow and keras (currently not other backend are suppoted rather
than `tf`), the tested version are listed below:

- python:               3.6.7
- tensorflow:           1.12.0
- tensorflow-base:      1.12.0
- tensorflow-gpu:       1.12.0
- keras:                2.2.4
- keras-applications:   1.0.6
- keras-base:           2.2.4
- keras-preprocessing:  1.0.5

## Run

Before you run, you need to speicify the dataset directory in your local machine. Go to the `reid.py` file,
check the global variable named `g_data_root`. If you don't have the dataset yet, you can get the dataset by
using the srcipt in the `datasets` folder. If you do so, set `g_data_root = './datasets'`.

To train or try the model out, go to the very end of the `reid.py` file. Comment the method you don't want
and uncomment the method you want then launch the terminal and run:

```
python reid.py
```

## Theory

For the theory behind the code, please check with the wiki.
