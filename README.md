# Machine Learning hacks written in Torch7

This repository contains a few **Machine Learning** hacks written in **Torch7**
(or Julia). Most of the scripts implement basic machine learning **algorithms**
or follow proofs from Christopher Bishop's "Pattern Recognition and Machine
Learning".

The biggest part of the code is written in lua modules, but some proofs are
written in iTorch notebooks.

The code is written in weekends so there's a slow pace in adding new algorithms.

## What can you find here

 - a very flexible module that generates synthetic data sets for you

    ds = require('synthetic_data_sets.lua')
    ds.random_data()

    params = {['D'] = 2, ['K'] = 2, ['uniform'] = true}
    ds.random_data(params)
