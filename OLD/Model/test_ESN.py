# -*- coding: utf-8 -*- 
# @Time : 2022/2/13 23:30 
# @Author : lepold
# @File : test_ESN.py


import numpy as np
import unittest
import matplotlib.pyplot as plt
from ESN import ESN


class Lorenz63():
    """
      It's a differential equations:
          dx/dt=s(y-x)
          dy/dt=rx-y-xz
          dz/dt=xy-bz
      parameters:[s, r, b]
      Given xt-1, yt-1, zt-1 and then calculate the state of the next time step.
    """

    def __init__(self, s, r, b, delta_t=0.01):
        self.s = s
        self.r = r
        self.b = b
        self.delta_t = delta_t
        self.x, self.y, self.z = 0., 1., 1.05

    @property
    def state(self):
        return np.array([self.x, self.y, self.z])

    def update(self, dx, dy, dz):
        # f = self.__getattribute__(f_str)
        # if f is None:
        #     self.__setattr__(f_str, df * self.delta_t)
        # else:
        #     f += df * self.delta_t
        self.x = self.x + dx
        self.y = self.y + dy
        self.z = self.z + dz

    def run(self):
        dx = (self.s * (self.y - self.x)) * self.delta_t
        dy = (self.r * self.x - self.y - self.x * self.z) * self.delta_t
        dz = (self.x * self.y - self.b * self.z) * self.delta_t
        self.update(dx, dy, dz)
        return np.array([self.x, self.y, self.z])

    def res(self, time, *args):
        if len(args) == 0:
            self.x, self.y, self.z = np.random.rand(3)
        elif len(args) == 1:
            self.x, self.y, self.z = np.random.RandomState(args).rand(3)
        else:
            self.x, self.y, self.z = args

        states = []
        for i in range(time):
            state = self.run()
            states.append(state)
        states = np.stack(states, axis=1)
        washout = int(time / 100)
        washout = washout if washout < 100 else 100
        states = states[:, washout:]
        return states

class TestCase(unittest.TestCase):
    @staticmethod
    def loss(x, y, dim=1):
        loss = np.linalg.norm(x - y, ord=2, axis=dim)
        return loss
    def test_lr_esn(self):
        dt = 0.02
        lorenz = Lorenz63(10., 28., 8 / 3, delta_t=dt)
        u = lorenz.res(20100, 2).T
        u_test = lorenz.res(20100, 3).T
        eta = 0.2
        seq_len, D = u.shape
        train_inputs, train_targets = u[:-1, :], u[1:, :]
        test_inputs, test_targets = u_test[:-1, :], u_test[1:, :]
        # noise_data = u + np.random.multivariate_normal(np.zeros(D), cov=np.eye(D) * eta, size=seq_len)
        esn = ESN(n_inputs=3,
                  n_outputs=3,
                  input_scaling=1 / 50,
                  teacher_scaling=1 / 50,
                  n_reservoir=88,
                  spectral_radius=0.81,
                  leaky_rate=0.6,
                  random_state=None,
                  noise=0.,
                  sparsity=0.4,
                  extended_states=False,
                  inverse="cholesky",
                  silent=True)
        esn.fit(train_inputs, train_targets)
        time = 500
        lyapunov_exp = 0.91
        u_predict = esn.predict(test_inputs[0], seq_len=time, reset_state=False)
        max_time_unit = int(time * dt / lyapunov_exp)
        time_ticks = [l / lyapunov_exp / dt for l in range(max_time_unit)]

        fig, ax = plt.subplots(4, 1, figsize=(8, 6))
        ax = ax.flatten()
        coords = ["x coordinate", "y coordinate", "z coordinate"]
        loss = self.loss(test_targets[:time, :], u_predict, dim=1)
        for i in range(3):
            ax[i].plot(range(time), u_test[:time, i], 'k', lw=1, label="target system")
            ax[i].plot(range(time), u_predict[:, i], 'r', lw=1, label="free running RFDA")
            ax[i].set_xticks([])
            ax[i].text(0.1, 0.8, coords[i], fontsize=10, ha='center', va='center', color='b', transform=ax[i].transAxes)
        ax[3].plot(range(time), loss, 'k', lw=1, label="loss")
        ax[3].legend(loc=(0.05, 0.7), fontsize=12)
        ax[0].legend(loc=(0.61, 1.1), fontsize='x-small')
        ax[3].set_xticks(time_ticks)
        ax[3].set_xticklabels(np.arange(max_time_unit))
        ax[3].set_xlabel('$ \lambda_{max}t $')
        ax[0].set_title("LR_without noise")
        plt.savefig("./ESNLR_lorenz.png")
        plt.show()