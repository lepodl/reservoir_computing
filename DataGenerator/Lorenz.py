# -*- coding: utf-8 -*- 
# @Time : 2022/2/28 17:04 
# @Author : lepold
# @File : Lorenz.py

import numpy as np


class Lorenz(object):
    """
     It's a differential equations:
         dx/dt=s(y-x)
         dy/dt=rx-y-xz
         dz/dt=xy-bz
     parameters:[s, r, b]
     Given xt-1, yt-1, zt-1 and then calculate the state of the next time step.
    """

    @staticmethod
    def RK4(t, r, f, dt):
        k1 = dt * f(t, r)
        k2 = dt * f(t + dt / 2, r + k1 / 2)
        k3 = dt * f(t + dt / 2, r + k2 / 2)
        k4 = dt * f(t + dt, r + k3)
        return r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def __init__(self, s, r, b, delta_t, *args):
        self.s = s
        self.r = r
        self.b = b
        self.delta_t = delta_t
        if len(args) == 0:
            self.state = np.random.rand(3)
        elif len(args) == 1:
            self.state = np.random.RandomState(args).rand(3)
        else:
            self.state = args

    def func(self, t, r):
        dx = (self.s * (r[1] - r[0]))
        dy = (self.r * r[0] - r[1] - r[0] * r[2])
        dz = (r[0] * r[1] - self.b * r[2])
        return np.array([dx, dy, dz])

    def propagate(self, time_train, time_washout=10, require_detail=False, noise=0.2):
        n = int(time_train / self.delta_t)
        n_washout = int(time_washout / self.delta_t)
        t = 0.
        out = np.empty((3, n, 4))
        out[:, 0, 0] = self.state
        if noise is None:
            print('no inner noise')
            for i in range(1, n):
                k1 = self.delta_t * self.func(t, self.state)
                k2 = self.delta_t * self.func(t + self.delta_t / 2, self.state + k1 / 2)
                k3 = self.delta_t * self.func(t + self.delta_t / 2, self.state + k2 / 2)
                k4 = self.delta_t * self.func(t + self.delta_t, self.state + k3)
                out[:, i, 0] = self.state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                out[:, i-1, 1:4] = np.stack([self.state + k1 / 2, self.state + k2 / 2, self.state + k3], axis=-1)
                self.state = out[:, i, 0]
                t += self.delta_t
        else:
            print('with inner noise')
            for i in range(1, n):
                pertubation = np.random.normal(size=(3,)) * np.sqrt(noise)
                k1 = self.delta_t * (self.func(t, self.state) + pertubation)
                k2 = self.delta_t * (self.func(t + self.delta_t / 2, self.state + k1 / 2) + pertubation)
                k3 = self.delta_t * (self.func(t + self.delta_t / 2, self.state + k2 / 2) + pertubation)
                k4 = self.delta_t * (self.func(t + self.delta_t, self.state + k3) + pertubation)
                out[:, i, 0] = self.state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                out[:, i-1, 1:4] = np.stack([self.state + k1 / 2, self.state + k2 / 2, self.state + k3], axis=-1)
                self.state = out[:, i, 0]
                t += self.delta_t
        if not require_detail:
            return out[:, n_washout:, 0]
        else:
            return out[:, n_washout:, :]