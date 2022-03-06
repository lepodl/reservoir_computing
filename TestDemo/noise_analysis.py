# -*- coding: utf-8 -*- 
# @Time : 2022/3/1 21:43 
# @Author : lepold
# @File : noise_analysis.py

import os
import numpy as np
from Model.Esn import Esn
from DataGenerator.Lorenz import Lorenz
from multiprocessing.pool import Pool as pool
from mpi4py import MPI


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


def run_lr(arg):
    idx, eta, min_hidden, min_radius = arg
    dt = 0.01
    train_len = 1000
    model = Lorenz(10., 28., 8 / 3, dt, idx)
    states = model.propagate(80, 10)
    train_data = states[:, :train_len]
    test_data = states[:, train_len:(train_len + 1000)]
    train_data_noise = train_data + np.random.multivariate_normal(np.zeros(3), np.eye(3) * eta, size=train_len).T
    esn = Esn(n_inputs=3,
              n_outputs=3,
              n_reservoir=int(min_hidden),
              leaky_rate=0.987,
              spectral_radius=min_radius,
              random_state=idx,
              sparsity=0.2,
              silent=True,
              ridge_param=1e-5)
    test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
    test_length = 800
    assert test_targets.shape[1] > test_length
    test_targets = test_targets[:, :test_length]
    esn.fit(train_data_noise, None)
    # esn.fit_da(train_data_noise, None, ensembles=300, eta=eta, gamma=1000., initial_zero=False)
    prediction = esn.forward(test_inputs[:, 0], n_iteration=test_length)
    power = prediction_power(prediction, test_targets)
    return power


def run_da(arg):
    idx, eta, min_hidden, min_radius = arg
    dt = 0.01
    train_len = 1000
    model = Lorenz(10., 28., 8 / 3, dt, idx)
    states = model.propagate(80, 10)
    train_data = states[:, :train_len]
    test_data = states[:, train_len:(train_len + 1000)]
    train_data_noise = train_data + np.random.multivariate_normal(np.zeros(3), np.eye(3) * eta, size=train_len).T
    esn = Esn(n_inputs=3,
              n_outputs=3,
              n_reservoir=int(min_hidden),
              leaky_rate=0.987,
              spectral_radius=min_radius,
              random_state=idx,
              sparsity=0.4,
              silent=True,
              ridge_param=1e-5)
    test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
    test_length = 800
    assert test_targets.shape[1] > test_length
    test_targets = test_targets[:, :test_length]
    # esn.fit(train_data_noise, None)
    esn.fit_da(train_data_noise, None, ensembles=300, eta=eta, gamma=1000., initial_zero=False)
    prediction = esn.forward(test_inputs[:, 0], n_iteration=test_length)
    power = prediction_power(prediction, test_targets)
    return power


# glbal parameter


etas = np.linspace(-5, 5, 50)
etas = np.exp(etas)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
os.makedirs("../Data/res_lr", exist_ok=True)
os.makedirs("../Data/res_da", exist_ok=True)

for i in range(rank, 50, size):
    eta = etas[i]
    with pool(processes=50) as p:
        res_lr = p.map(run_lr, zip(np.random.randint(0, 1000, 100), np.ones(100, dtype=np.float32) * eta,
                                   np.ones(100, dtype=np.int32) * 92,
                                   np.ones(100, dtype=np.float32) * 0.148))
        res_da = p.map(run_da, zip(np.random.randint(0, 1000, 100), np.ones(100, dtype=np.float32) * eta,
                                   np.ones(100, dtype=np.int32) * 92,
                                   np.ones(100, dtype=np.float32) * 0.148))
    res_lr = np.array(res_lr)
    res_da = np.array(res_da)
    np.save(f"../Data/res_lr/{i}_power.npy", res_lr)
    np.save(f"../Data/res_da/{i}_power.npy", res_da)
print("DONE!")
