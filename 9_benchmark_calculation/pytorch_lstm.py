#!/usr/bin/env python

import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import timeit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

times = {}
seq_len = 4
feature_size = 4
num_evals = 100
num_repeats = 10
layer_sizes = (2,3,4)
for num_layers in layer_sizes:
    for batch_size in range(1, 8):
        net = nn.LSTM(feature_size, feature_size, num_layers, bias=False)
        x = torch.randn(seq_len, batch_size, feature_size)
        loss = net.forward(x)[0].sum()
       
        T = timeit.Timer('loss.backward(retain_graph=True)', setup='from __main__ import loss')
        t = min(T.repeat(repeat=num_repeats,number=num_evals))
        times[(num_layers, feature_size, seq_len, batch_size)] = t/num_evals
        print('[%d, %d]: %.2f us'%(num_layers, batch_size, t*1e6/num_evals))

models = {}
for num_layers in layer_sizes:
    ts = [times[(num_layers, feature_size, seq_len, batch_size)] for batch_size in range(1,8)]

    model = LinearRegression()
    model.fit(np.array(range(1,8)).reshape(-1, 1), np.array(ts).reshape(-1, 1))
    models[num_layers] = model

print("Estimated overheads: ")
for num_layers in layer_sizes:
    print("  [%d]: %.2f us"%(num_layers, models[num_layers].intercept_*1e6/(num_layers*seq_len)))
