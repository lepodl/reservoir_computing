# -*- coding: utf-8 -*- 
# @Time : 2022/3/1 13:26 
# @Author : lepold
# @File : compose_data.py

import os
import numpy as np
import pandas as pd


def compose_dependence_analysis():
    data_dir = "../Data/split_data2"
    leaky_range = [0.1, 1.]
    spectral_range = [0.5, 1.5]
    num_samples = 100

    leaks = np.linspace(leaky_range[0], leaky_range[1], num=num_samples)
    spectrals = np.linspace(spectral_range[0], spectral_range[1], num=num_samples)

    samples = 100
    total_length = num_samples * num_samples
    res = np.zeros((total_length, samples))
    for i in range(total_length):
        out = np.load(os.path.join(data_dir, f"{i}.npy"))
        assert out.shape[0] == samples
        res[i] = out
    res = res.reshape((num_samples, num_samples, samples))
    np.savez("../Data/dependece_leak_spectral_da.npz", res=res, leaks=leaks, spectrals=spectrals)


def compose_noise_analysis():
    etas = np.linspace(-5, 5, 50)
    powers = []

    for i in range(50):
        power = np.load(f"../Data/res_lr/{i}_power.npy")
        powers.append(power)
    powers_dict = dict(zip(etas, powers))
    df = pd.DataFrame([powers_dict])

    df.to_csv("../Data/noise_dependence_esnlr.csv")

    powers = []
    for i in range(50):
        power = np.load(f"../Data/res_da/{i}_power.npy")
        powers.append(power)
    powers_dict = dict(zip(etas, powers))
    df = pd.DataFrame([powers_dict])

    df.to_csv("../Data/noise_dependence_esnda_biased.csv")
    print("DONE")


if __name__ == '__main__':
    compose_dependence_analysis()
    # compose_noise_analysis()
