# -*- coding: utf-8 -*- 
# @Time : 2022/2/21 11:31 
# @Author : lepold
# @File : ESNDA_compare.py
import os

import matplotlib.pyplot as plt
import numpy as np
from ESN import Esn
from lorenz import lorenz
from multiprocessing.pool import Pool as pool
from mpi4py import MPI


def pred_power(x, y, threshold=10):
    loss = np.linalg.norm(x - y, ord=2, axis=0)
    initial = loss[:20]
    if (initial > 3).all():
        index = 0
    else:
        try:
            index = np.where(loss>threshold)[0][0]
        except:
            index = x.shape[0]
    return index


def mse(outputs, targets):
    return np.mean(np.sqrt(np.sum((targets - outputs) ** 2, axis=0)))

def random_grid_search(traindata, testdata, hidden_range, radius_range, num_samples):
    train_inputs = traindata
    test_inputs, test_targets = testdata[:, :], testdata[:, 1:]
    test_length = 700
    assert test_targets.shape[1]>test_length
    test_targets = test_targets[:, :test_length]

    hidden = np.random.randint(hidden_range[0], hidden_range[1], size=num_samples, dtype=np.int32)
    radius = np.random.rand(num_samples) * (radius_range[1] - radius_range[0]) + radius_range[0]

    min_error = None
    min_hidden, min_radius = None, None
    # mses = []

    for hid, rad in zip(hidden, radius):
        esn = Esn(n_inputs=3,
                  n_outputs=3,
                  n_reservoir=int(hid),
                  leaky_rate=1.,
                  spectral_radius=rad,
                  random_state=None,
                  sparsity=0.2,
                  silent=True,
                  ridge_param=1e-5)
        esn.fit(train_inputs)

        prediction = esn.forward(test_inputs[:, 0], n_iteration=test_length)
        error = mse(test_targets, prediction)
        if not min_error or error < min_error:
            min_error, min_hidden, min_radius = error, hid, rad
        # mses.append(error)

    return min_error, min_hidden, min_radius


def run_lr(arg):
    idx, eta, min_hidden, min_radius = arg
    esn = Esn(n_inputs=3,
              n_outputs=3,
              n_reservoir=int(min_hidden),
              leaky_rate=1.,
              spectral_radius=min_radius,
              random_state=idx,
              sparsity=0.2,
              silent=True,
              ridge_param=1e-5)
    test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
    test_length = 800
    assert test_targets.shape[1]>test_length
    test_targets = test_targets[:, :test_length]
    esn.fit(train_data_noise, None)
    # esn.fit_da(train_data_noise, None, ensembles=300, eta=eta, gamma=1000., initial_zero=False)
    prediction = esn.forward(test_inputs[:, 0], n_iteration=test_length)
    power = pred_power(prediction, test_targets)
    return power

def run_da(arg):
    idx, eta, min_hidden, min_radius = arg
    esn = Esn(n_inputs=3,
              n_outputs=3,
              n_reservoir=int(min_hidden),
              leaky_rate=1.,
              spectral_radius=min_radius,
              random_state=idx,
              sparsity=0.2,
              silent=True,
              ridge_param=1e-5)
    test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
    test_length = 800
    assert test_targets.shape[1]>test_length
    test_targets = test_targets[:, :test_length]
    # esn.fit(train_data_noise, None)
    esn.fit_da(train_data_noise, None, ensembles=400, eta=eta, gamma=1000., initial_zero=False)
    prediction = esn.forward(test_inputs[:, 0], n_iteration=test_length)
    power = pred_power(prediction, test_targets)
    return power



# glbal parameter
dt = 0.01
train_len = 4000
model = lorenz(10., 28., 8 / 3, dt)
states = model.propagate(80, 10)
train_data = states[:, :train_len]
test_data = states[:, train_len:]

etas = np.linspace(-5, 5, 50)
etas = np.power(10, etas)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
os.makedirs("result/res_lr", exist_ok=True)
os.makedirs("result/res_da", exist_ok=True)

for i in range(rank, 50, size):
    eta = etas[i]
    train_data_noise = train_data + np.random.multivariate_normal(np.zeros(3), np.eye(3) * eta, size=train_len).T
    error, min_hidden, min_radius, = random_grid_search(train_data_noise, test_data, hidden_range=[100, 200],
                                                    radius_range=[0.1, 1.2], num_samples=200)
    print(f"random grid search for noise eta {i} is {error:.2f}")
    with pool(processes=50) as p:
        res_lr = p.map(run_lr, zip(np.arange(100), np.ones(100, dtype=np.float32) * eta, np.ones(100, dtype=np.int32) * min_hidden, np.ones(100, dtype=np.float32) * min_radius))
        res_da = p.map(run_da, zip(np.arange(100), np.ones(100, dtype=np.float32) * eta, np.ones(100, dtype=np.int32) * min_hidden, np.ones(100, dtype=np.float32) * min_radius))
    res_lr = np.array(res_lr)
    res_da = np.array(res_da)
    np.save(f"result/res_lr/{i}_power.npy", res_lr)
    np.save(f"result/res_da/{i}_power.npy", res_da)
print("DONE!")


