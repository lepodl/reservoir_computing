# -*- coding: utf-8 -*- 
# @Time : 2022/2/22 15:09 
# @Author : lepold
# @File : infer_bifuraction.py

import matplotlib.pyplot as plt
import numpy as np
from lorenz import lorenz
from Model.ESN_Control_Euler import Esn
from utils.utils import downsample_curvature


base_num = 0
elev, azim = [15, 100]  # Azimuth 方位角， elevation 仰角
nF = 200
color = plt.get_cmap("winter")
winter = color(np.linspace(0, 1, int(1.2 * nF)))


def fixed_point(rho):
    return np.array([np.sqrt(8 / 3 * (rho - 1)), np.sqrt(8 / 3 * (rho - 1)), rho - 1])

def plot(data, contorl_par, index, name):
    global nF
    global winter
    global elev
    global azim
    global base_num
    fig = plt.figure(base_num, figsize=(5, 5), dpi=200)
    base_num += 1
    ax = fig.gca(projection='3d')
    ll = data.shape[1]
    chunks = np.floor(np.linspace(0, ll, nF)).astype(np.int)
    for i in range(1, nF):
        col = winter[contorl_par[index[chunks[i - 1]]]]
        ax.plot(data[0, chunks[i - 1]:chunks[i]], data[1, chunks[i - 1]:chunks[i]], data[2, chunks[i - 1]:chunks[i]],
                lw=0.2, color=col)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    # ax.set_title("Lorenz Attractor")
    ax.grid(False)
    ax.view_init(elev=elev,  # 仰角
                 azim=azim,  # 方位角
                 )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.axis('off')
    ax.xaxis.pane.fill = False  # Left pane
    ax.yaxis.pane.fill = False  # Right pane
    # print('ax.elev {}'.format(ax.elev))  # default 30
    # print('ax.azim {}'.format(ax.azim))  # default -60
    # fig.savefig(os.path.join("fig", name + ".png"))
    return fig


def base_plot(data):
    global base_num
    fig = plt.figure(base_num, figsize=(5, 5), dpi=200)
    base_num += 1
    ax = fig.gca(projection='3d')
    states, ind = downsample_curvature(data, 0.2, np.array([100, 15]))

    ax.plot(states[0, :], states[1, :], states[2, :], lw=1.)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    # ax.set_title("Lorenz Attractor")
    ax.grid(False)
    ax.view_init(elev=15,  # 仰角
                 azim=100,  # 方位角
                 )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.axis('off')
    ax.xaxis.pane.fill = False  # Left pane
    ax.yaxis.pane.fill = False  # Right pane
    print('ax.elev {}'.format(ax.elev))  # default 30
    print('ax.azim {}'.format(ax.azim))  # default -60
    return fig

def version1():
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
    total_data = np.zeros((3, n * 4))
    t_ind = np.arange(n_waste, n)
    t_ind = np.concatenate([t_ind, t_ind + n, t_ind + 2 * n, t_ind + 3 * n])
    print("\n Simulating lorenz")
    model = lorenz(10., rhos[0], 8 / 3, delta_t, *(fixed_point(rhos[0]) + scale[0] * pertubation))
    total_data[:, :n] = model.propagate(t_train + t_waste, t_washout)
    fig0 = base_plot(total_data[:, n_waste:n])
    # downsample_data, ind = downsample_curvature(total_data[:, :n, 0], 0.2, np.array([100, 15]))
    print("| Progress: [{0:10s}] {1:.1f}%".format('#' * int(0.25 * 10), 0.25 * 100))
    model = lorenz(10., rhos[1], 8 / 3, delta_t, *(fixed_point(rhos[1]) + scale[1] * pertubation))
    total_data[:, n:2 * n] = model.propagate(t_train + t_waste, t_washout)
    fig1 = base_plot(total_data[:, n+n_waste:2 * n])
    print("| Progress: [{0:10s}] {1:.1f}%".format('#' * int(0.5 * 10), 0.5 * 100))
    model = lorenz(10., rhos[0], 8 / 3, delta_t, *(fixed_point(rhos[0]) - scale[0] * pertubation))
    total_data[:, 2 * n:3 * n] = model.propagate(t_train + t_waste, t_washout)
    fig2 = base_plot(total_data[:, 2 * n + n_waste:n * 3])
    print("| Progress: [{0:10s}] {1:.1f}%".format('#' * int(0.75 * 10), 0.75 * 100))
    model = lorenz(10., rhos[1], 8 / 3, delta_t, *(fixed_point(rhos[1]) - scale[1] * pertubation))
    total_data[:, -n:] = model.propagate(t_train + t_waste, t_washout)
    fig3 = base_plot(total_data[:, -n+n_waste:])
    print("| Progress: [{0:10s}] {1:.1f}%".format('#' * int(1. * 10), 1. * 100))
    eta = 0.1
    total_data = total_data + np.random.multivariate_normal(np.zeros(3), cov=np.eye(3) * eta, size=4 * n).T

    # reservoir states
    esn = Esn(n_inputs=3,
              n_reservoir=200,
              n_outputs=3,
              leaky_rate=1.,
              spectral_radius=0.95,
              random_state=None,
              sparsity=0.1,
              ridge_param=1e-2,
              delta_t=delta_t,
              silent=False,
              washout=2000,
              valid_index=t_ind)
    control_par = np.ones((1, n))
    control_par = np.concatenate([control_par * 0, control_par * 1, control_par * 0, control_par * 1], axis=1)
    print("\nSimulating and fitting reservoir")
    # esn.fit(total_data, control_par)
    esn.fit_da(total_data, control_par, ensembles=400, eta=eta, initial_zero=False)

    nT = 6000
    n_forward = 9000
    n_backward = 15000
    control_par = np.concatenate(
        [np.linspace(0, 5, nT), 5 * np.ones(n_forward), np.linspace(5, 0, nT), 0 * np.ones(n_backward)])
    control_par = control_par[np.newaxis, :, ]
    print("\nPredicting reservoir")
    output = esn.forward(control_par)

    prediction1 = output[:, :(nT + n_forward)]
    prediction2 = output

    print("\n Plot result")


    print('ax.elev {}'.format(elev))  # default 30
    print('ax.azim {}'.format(azim))  # default -60
    downsample_pred1, ind1 = downsample_curvature(prediction1, 0.1, np.array([azim, elev]))
    downsample_pred2, ind2 = downsample_curvature(prediction2, 0.1, np.array([azim, elev]))

    control_par1 = control_par[:, :(nT + n_forward)].squeeze()
    control_par2 = control_par[:, :].squeeze()

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

    fig4 = plot(downsample_pred1, cp1, ind1, "forward_bifuraction_noise")
    fig5 = plot(downsample_pred2, cp2, ind2, "backward_bifuraction_noise")
    plt.show()


def version2():
    pertubation = np.array([0.6, 1.1, 0.])
    scale = np.array([4., 2.8])
    rhos = np.linspace(23, 24, 4)

    delta_t = 0.01
    t_train = 200
    t_waste = 10
    n_train = int(t_train / delta_t)
    n_waste = int(t_waste / delta_t)
    n = n_train
    total_data = np.zeros((3, n_train * 4))
    t_ind = np.arange(n_waste, n)
    t_ind = np.concatenate([t_ind, t_ind + n, t_ind + 2 * n, t_ind + 3 * n])
    print("\n Simulating lorenz")
    model = lorenz(10., rhos[0], 8 / 3, delta_t, *(fixed_point(rhos[0]) + scale[0] * pertubation))
    total_data[:, :n] = model.propagate(t_train + t_waste, t_waste)
    fig0 = base_plot(total_data[:, :n])
    print("| Progress: [{0:10s}] {1:.1f}%".format('#' * int(0.25 * 10), 0.25 * 100))
    model = lorenz(10., rhos[1], 8 / 3, delta_t, *(fixed_point(rhos[1]) + scale[1] * pertubation))
    total_data[:, n:2 * n] = model.propagate(t_train + t_waste, t_waste)
    fig1 = base_plot(total_data[:, n:2 * n])
    print("| Progress: [{0:10s}] {1:.1f}%".format('#' * int(0.5 * 10), 0.5 * 100))
    model = lorenz(10., rhos[2], 8 / 3, delta_t, *(fixed_point(rhos[2]) - scale[0] * pertubation))
    total_data[:, 2 * n:3 * n] = model.propagate(t_train + t_waste, t_waste)
    fig2 = base_plot(total_data[:, 2*n:3*n])
    print("| Progress: [{0:10s}] {1:.1f}%".format('#' * int(0.75 * 10), 0.75 * 100))
    model = lorenz(10., rhos[3], 8 / 3, delta_t, *(fixed_point(rhos[3]) - scale[1] * pertubation))
    total_data[:, -n:] = model.propagate(t_train + t_waste, t_waste)
    fig3 = base_plot(total_data[:, -n:])
    print("| Progress: [{0:10s}] {1:.1f}%".format('#' * int(1. * 10), 1. * 100))
    eta = 0.01
    # total_data = total_data + np.random.multivariate_normal(np.zeros(3), cov=np.eye(3) * eta, size=4 * n).T

    # reservoir states
    esn = Esn(n_inputs=3,
              n_reservoir=200,
              n_outputs=3,
              leaky_rate=0.95,
              spectral_radius=0.95,
              random_state=None,
              sparsity=0.1,
              ridge_param=1e-5,
              delta_t=delta_t,
              silent=False,
              washout=500,
              valid_index=None)
    control_par = np.ones((1, n))
    control_par = np.concatenate([control_par * 0, control_par * 0.25, control_par * 0.5, control_par * 1.], axis=1)
    print("\nSimulating and fitting reservoir")
    esn.fit(total_data, control_par)
    # esn.fit_da(total_data, control_par, ensembles=500, eta=eta, initial_zero=False)

    nT = 6000
    n_forward = 10000
    n_backward = 12000
    control_par = np.concatenate(
        [np.linspace(0, 5, nT), 5 * np.ones(n_forward), np.linspace(5, 0, nT), 0 * np.ones(n_backward)])
    control_par = control_par[np.newaxis, :, ]
    print("\nPredicting reservoir")
    output = esn.forward(control_par)

    prediction1 = output[:, :(nT + n_forward)]
    prediction2 = output

    print("\n Plot result")
    print('ax.elev {}'.format(elev))  # default 30
    print('ax.azim {}'.format(azim))  # default -60
    downsample_pred1, ind1 = downsample_curvature(prediction1, 0.1, np.array([azim, elev]))
    downsample_pred2, ind2 = downsample_curvature(prediction2, 0.1, np.array([azim, elev]))

    control_par1 = control_par[:, :(nT + n_forward)].squeeze()
    control_par2 = control_par[:, :].squeeze()

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

    fig5 = plot(downsample_pred1, cp1, ind1, "forward_bifuraction_noise")
    fig6 = plot(downsample_pred2, cp2, ind2, "backward_bifuraction_noise")
    plt.show()

if __name__ == '__main__':
    version1()
