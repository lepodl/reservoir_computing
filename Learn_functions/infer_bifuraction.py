# -*- coding: utf-8 -*- 
# @Time : 2022/2/22 15:09 
# @Author : lepold
# @File : infer_bifuraction.py

import os
import matplotlib.pyplot as plt
import numpy as np
from lorenz import lorenz
from ESN_Control import Esn
from utils import downsample_curvature


def fixed_point(rho):
    return np.array([np.sqrt(8 / 3 * (rho - 1)), np.sqrt(8 / 3 * (rho - 1)), rho - 1])

def plot(data, contorl_par, index, name):
    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = fig.gca(projection='3d')
    ll = data.shape[1]
    chunks = np.floor(np.linspace(0, ll, nF)).astype(np.int)
    for i in range(1, nF):
        col = winter[contorl_par[index[chunks[i - 1]]]]
        ax.plot(data[0, chunks[i - 1]:chunks[i]], data[1, chunks[i - 1]:chunks[i]], data[2, chunks[i - 1]:chunks[i]],
                lw=0.2, color=col)
    # ax.set_xlabel("X Axis")
    # ax.set_ylabel("Y Axis")
    # ax.set_zlabel("Z Axis")
    # ax.set_title("Lorenz Attractor")
    ax.grid(False)
    ax.view_init(elev=elev,  # 仰角
                 azim=azim,  # 方位角
                 )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis('off')
    ax.xaxis.pane.fill = False  # Left pane
    ax.yaxis.pane.fill = False  # Right pane
    # print('ax.elev {}'.format(ax.elev))  # default 30
    # print('ax.azim {}'.format(ax.azim))  # default -60
    fig.savefig(os.path.join("fig", name + ".png"))


pertubation = np.array([0.6, 1.1, 0.])
scale = np.array([4., 2.8])
rhos = np.linspace(23, 24, 2)

delta_t = 0.01
t_train = 200
t_washout = 0
t_waste = 20
n_train = int(t_train / delta_t)
n_waste = int(t_waste / delta_t)
n = n_train + n_waste
total_data = np.zeros((3, n * 4, 4))
t_ind = np.arange(n_waste, n)
t_ind = np.concatenate([t_ind, t_ind + n, t_ind + 2 * n, t_ind + 3 * n])
print("\n Simulating lorenz")
model = lorenz(10., rhos[0], 8 / 3, delta_t, *(fixed_point(rhos[0]) + scale[0] * pertubation))
total_data[:, :n] = model.propagate(t_train + t_waste, t_washout)
# downsample_data, ind = downsample_curvature(total_data[:, :n, 0], 0.2, np.array([100, 15]))
print("| Progress: [{0:10s}] {1:.1f}%".format('#' * int(0.25 * 10), 0.25 * 100))
model = lorenz(10., rhos[1], 8 / 3, delta_t, *(fixed_point(rhos[1]) + scale[1] * pertubation))
total_data[:, n:2 * n] = model.propagate(t_train + t_waste, t_washout)
print("| Progress: [{0:10s}] {1:.1f}%".format('#' * int(0.5 * 10), 0.5 * 100))
model = lorenz(10., rhos[0], 8 / 3, delta_t, *(fixed_point(rhos[0]) - scale[0] * pertubation))
total_data[:, 2 * n:3 * n] = model.propagate(t_train + t_waste, t_washout)
print("| Progress: [{0:10s}] {1:.1f}%".format('#' * int(0.75 * 10), 0.75 * 100))
model = lorenz(10., rhos[1], 8 / 3, delta_t, *(fixed_point(rhos[1]) - scale[1] * pertubation))
total_data[:, -n:] = model.propagate(t_train + t_waste, t_washout)
print("| Progress: [{0:10s}] {1:.1f}%".format('#' * int(1. * 10), 1. * 100))

eta = 0.1
noise = np.random.multivariate_normal(np.zeros(3), np.eye(3)*eta, size=n*4).T
noise_data = total_data + noise[:, :, np.newaxis]

# reservoir states
esn = Esn(n_inputs=3,
          n_reservoir=450,
          n_outputs=3,
          gamma=100,
          spectral_radius=0.95,
          random_state=None,
          sparsity=0.1,
          ridge_param=1e-7,
          delta_t=delta_t,
          silent=False,
          valid_index=t_ind)
control_par = np.ones((1, n, 4))
control_par = np.concatenate([control_par * 0, control_par * 1, control_par * 0, control_par * 1], axis=1)
print("\nSimulating and fitting reservoir")
esn.fit(noise_data, control_par)

nT = 6000
n_forward = 9000
n_backward = 15000
control_par = np.concatenate(
    [np.linspace(0, 5, nT), 5 * np.ones(n_forward), np.linspace(5, 0, nT), 0 * np.ones(n_backward)])
Diff = np.diff(control_par, 1, axis=0)
Diff = np.concatenate([Diff, np.array([0])])
control_par = np.stack([control_par, control_par + Diff / 2, control_par + Diff / 2, control_par + Diff], axis=1)
control_par = control_par[np.newaxis, :, :]
print("\nPredicting reservoir")
output = esn.forward(control_par)

prediction1 = output[:, :(nT + n_forward)]
prediction2 = output

print("\n Plot result")
color = plt.get_cmap("winter")
elev, azim = [15, 100]  # Azimuth 方位角， elevation 仰角
nF = 200
winter = color(np.linspace(0, 1, int(1.2 * nF)))
print('ax.elev {}'.format(elev))  # default 30
print('ax.azim {}'.format(azim))  # default -60
downsample_pred1, ind1 = downsample_curvature(prediction1, 0.05, np.array([azim, elev]))
downsample_pred2, ind2 = downsample_curvature(prediction2, 0.05, np.array([azim, elev]))

control_par1 = control_par[:, :(nT + n_forward), 0].squeeze()
control_par2 = control_par[:, :, 0].squeeze()

cp1 = np.floor((control_par1 - np.min(control_par1)) / (
            np.max(control_par1) - np.min(control_par1)) * (nF - 3) + 1)
cp2 = np.floor((control_par2 - np.min(control_par2)) / (
            np.max(control_par2) - np.min(control_par2)) * (nF - 3) + 1)


cp1 = cp1.astype(np.int)
cp2 = cp2.astype(np.int)


save_res = False
if save_res:
    np.savez("result/forward_predict_bifuraction.npz", data=downsample_pred1, index=ind1, c=control_par1)
    np.savez("result/backward_predict_bifuraction.npz", data=downsample_pred2, index=ind2, c=control_par2)

plot(downsample_pred1, cp1, ind1, "forward_bifuraction_noise")
plot(downsample_pred2, cp2, ind2, "backward_bifuraction_noise")
