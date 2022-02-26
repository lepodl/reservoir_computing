# -*- coding: utf-8 -*- 
# @Time : 2022/2/21 11:31 
# @Author : lepold
# @File : ESNDA_compare.py
import matplotlib.pyplot as plt
import numpy as np
from ESN import Esn
from lorenz import lorenz
import pandas as pd
from multiprocessing.pool import Pool as pool
from mpi4py import MPI


def pred_power(x, y, threshold=10):
    loss = np.linalg.norm(x - y, ord=2, axis=0)
    try:
        index = np.where(loss>threshold)[0][0]
    except:
        index = x.shape[0]
    return index

def run_enslr():
    dt = 0.01
    train_len = 4000
    model = lorenz(10., 28., 8 / 3, dt)
    states = model.propagate(80, 10)
    train_data = states[:, :train_len]
    test_data = states[:, train_len:]

    etas = np.linspace(-5, 1, 20)
    etas = np.exp(etas)
    power_dict = {}
    with tqdm(total=4000, desc='Processing', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
        for idx, eta in enumerate(etas):
            power_dict[eta] = []
            for _ in range(200):
                esn = Esn(n_inputs=3,
                          n_outputs=3,
                          n_reservoir=92,
                          leaky_rate=0.987,
                          spectral_radius=0.148,
                          random_state=None,
                          sparsity=0.4,
                          silent=True)
                train_data = train_data + np.random.multivariate_normal(np.zeros(3), np.eye(3) * eta, size=train_len).T
                test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
                test_length = 800
                assert test_targets.shape[1]>test_length
                test_targets = test_targets[:, :test_length]
                esn.fit(train_data, None)
                # esn.fit_da(train_data, None, ensembles=300, eta=0.2, gamma=1000., initial_zero=False)
                prediction = esn.forward(test_inputs[:, 0], n_iteration=test_length)
                power = pred_power(prediction, test_targets)
                power_dict[eta].append(power)
                pbar.update(1)
    df = pd.DataFrame(power_dict)
    df.to_csv("result/noise_dependence_enslr.csv")

def run_ensda_zero():
    dt = 0.01
    train_len = 1000
    model = lorenz(10., 28., 8 / 3, dt)
    states = model.propagate(80, 10)
    train_data = states[:, :train_len]
    test_data = states[:, train_len:]

    etas = np.linspace(-5, 1, 20)
    etas = np.exp(etas)
    power_dict = {}
    with tqdm(total=2000, desc='Processing', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
        for idx, eta in enumerate(etas):
            power_dict[eta] = []
            for _ in range(100):
                esn = Esn(n_inputs=3,
                          n_outputs=3,
                          n_reservoir=92,
                          leaky_rate=0.987,
                          spectral_radius=0.148,
                          random_state=None,
                          sparsity=0.4,
                          silent=True)
                train_data = train_data + np.random.multivariate_normal(np.zeros(3), np.eye(3) * eta, size=train_len).T
                test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
                test_length = 800
                assert test_targets.shape[1]>test_length
                test_targets = test_targets[:, :test_length]
                # esn.fit(train_data, None)
                esn.fit_da(train_data, None, ensembles=300, eta=eta, gamma=1000., initial_zero=True)
                prediction = esn.forward(test_inputs[:, 0], n_iteration=test_length)
                power = pred_power(prediction, test_targets)
                power_dict[eta].append(power)
                pbar.update(1)
    df = pd.DataFrame(power_dict)
    df.to_csv("result/noise_dependence_esnda_zero.csv")

def run_ensda_biased():
    dt = 0.01
    train_len = 4000
    model = lorenz(10., 28., 8 / 3, dt)
    states = model.propagate(80, 10)
    train_data = states[:, :train_len]
    test_data = states[:, train_len:]

    etas = np.linspace(-5, 1, 20)
    etas = np.exp(etas)
    power_dict = {}
    with tqdm(total=4000, desc='Processing', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
        for idx, eta in enumerate(etas):
            power_dict[eta] = []
            for _ in range(200):
                esn = Esn(n_inputs=3,
                          n_outputs=3,
                          n_reservoir=92,
                          leaky_rate=0.987,
                          spectral_radius=0.148,
                          random_state=None,
                          sparsity=0.4,
                          silent=True)
                train_data = train_data + np.random.multivariate_normal(np.zeros(3), np.eye(3) * eta, size=train_len).T
                test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
                test_length = 800
                assert test_targets.shape[1]>test_length
                test_targets = test_targets[:, :test_length]
                # esn.fit(train_data, None)
                esn.fit_da(train_data, None, ensembles=300, eta=eta, gamma=1000., initial_zero=False)
                prediction = esn.forward(test_inputs[:, 0], n_iteration=test_length)
                power = pred_power(prediction, test_targets)
                power_dict[eta].append(power)
                pbar.update(1)
    df = pd.DataFrame(power_dict)
    df.to_csv("result/noise_dependence_esnda_biased.csv")


def run(arg):
    idx, eta = arg
    global train_data
    esn = Esn(n_inputs=3,
              n_outputs=3,
              n_reservoir=92,
              leaky_rate=0.987,
              spectral_radius=0.148,
              random_state=idx,
              sparsity=0.4,
              silent=True)
    train_data_noise = train_data + np.random.multivariate_normal(np.zeros(3), np.eye(3) * eta, size=train_len).T
    test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
    test_length = 800
    assert test_targets.shape[1]>test_length
    test_targets = test_targets[:, :test_length]
    # esn.fit(train_data, None)
    esn.fit_da(train_data_noise, None, ensembles=300, eta=eta, gamma=1000., initial_zero=False)
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

etas = np.linspace(-5, 1, 20)
etas = np.exp(etas)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

for i in range(rank, 20, size):
    eta = etas[i]
    with pool(processes=50) as p:
        res = p.map(run, zip(range(100), np.ones(100) * eta))
    res = np.array(res)
    print(i, res[:5])
    np.save(f"npy_res/{i}_power.npy", res)
print("DONE!")

# power_dict = dict(zip(etas, zip(*res)))
# df = pd.DataFrame(power_dict)
# df.to_csv("result/noise_dependence_esnda_biased_test.csv")

