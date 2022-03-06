# -*- coding: utf-8 -*- 
# @Time : 2022/3/1 12:39 
# @Author : lepold
# @File : dependence_analysis.py
import os

import numpy as np
from time import time
from Model.Esn import Esn
from DataGenerator.Lorenz import Lorenz
from mpi4py import MPI
from multiprocessing.pool import Pool


def mse(outputs, targets):
    if outputs.size != targets.size:
        raise ValueError(u"Ouputs and targets ndarray don have the same number of elements")
    return np.mean(np.linalg.norm(outputs - targets, ord=2, axis=0))


def distance(x, y):
    loss = np.linalg.norm(x - y, ord=2, axis=0)
    loss = loss / np.linalg.norm(x, ord=2, axis=0)
    return loss


def prediction_power(x, y, dt=0.01, unit=0.91):
    loss = distance(x, y)
    if (loss[:10] < 0.1).all():
        try:
            out = np.where(loss > 0.55)[0][0]
        except:
            out = x.shape[1]
    else:
        out = 0
    out = out * dt * unit
    return out


def run(args):
    idx, leak, spectral, n_reservoir = args
    dt = 0.01
    model = Lorenz(10., 28., 8 / 3, dt, idx)
    states = model.propagate(50, 10)
    train_data = states[:, :1000]
    # model = Lorenz(10., 28., 8 / 3, dt)
    # states = model.propagate(50, 10)
    test_data = states[:, 1000:2000]
    del states

    test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
    test_length = test_targets.shape[1]

    eta = 1.
    noise_train_data = train_data + np.random.multivariate_normal(np.zeros(3), np.eye(3) * eta,
                                                                  size=train_data.shape[1]).T

    esn = Esn(n_inputs=3,
              n_outputs=3,
              n_reservoir=int(n_reservoir),
              leaky_rate=leak,
              spectral_radius=spectral,
              random_state=idx,
              sparsity=0.4,
              silent=False)

    # esn.fit(noise_train_data)
    esn.fit_da(noise_train_data, None, ensembles=300, eta=eta, gamma=1000., initial_zero=False)
    prediction = esn.forward(test_inputs[:, 0], n_iteration=test_length)
    pred_power = prediction_power(test_targets, prediction)
    return pred_power


leaky_range = [0.1, 1.]
spectral_range = [0.5, 1.5]
num_samples = 100
n_reservoir = 100
samples = 100

os.makedirs("../Data/split_data", exist_ok=True)
leaks = np.linspace(leaky_range[0], leaky_range[1], num=num_samples)
spectrals = np.linspace(spectral_range[0], spectral_range[1], num=num_samples)
xx, yy = np.meshgrid(leaks, spectrals)
xx = xx.reshape(-1)
yy = yy.reshape(-1)
total_length = xx.shape[0]

os.makedirs("../Data/split_data2", exist_ok=True)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


for i in range(rank, total_length, size - 1):
    print(f"rank {i} start")
    leak = xx[i]
    spectral = yy[i]
    with Pool(processes=50) as p:
        out = p.map(run, zip(np.random.randint(0, 1000, 100), np.ones(samples) * leak, np.ones(samples) * spectral,
                             np.ones(samples, dtype=np.int) * n_reservoir))
    out = np.array(out, dtype=np.float32)
    np.save(f"../Data/split_data2/{i}.npy", np.array(out))
    print(f"rank {i} end")

