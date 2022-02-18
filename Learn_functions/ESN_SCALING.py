# -*- coding: utf-8 -*- 
# @Time : 2022/2/16 14:28 
# @Author : lepold
# @File : ESN.py


import numpy as np
from lorenz import lorenz


class Esn(object):
    def __init__(self, n_inputs, n_outputs, n_reservoir=200, leaky_rate=0.1,
                 sparsity=0., random_state=None, spectral_radius=0.95, scaling=None,
                 ridge_param=1e-5, delta_t=0.01, washout=100, silent=True):

        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.leaky_rate = leaky_rate
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        self.delta_t = delta_t

        self.ridge_param = ridge_param
        self.random_state = random_state
        self.washout = washout
        self.scaling = scaling

        self.state = np.zeros(n_reservoir)
        self.W = None
        self.W_in = None
        self.W_bias = None
        self.W_lr = None

        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand
        self.silent = silent
        self.init_weights(sig=0.008)

    def init_weights(self, sig=0.008):
        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / radius)

        self.W_in = self.random_state_.rand(self.n_reservoir, self.n_inputs) * 2 - 1
        self.W_bias = self.random_state_.rand(self.n_reservoir) * 2 - 1

        # fixed point
        # r0 = self.random_state_.rand(self.n_reservoir) * 0.2 + 0.8 * np.sign(self.random_state_.rand(self.n_reservoir) - 0.5)
        # x0 = self.random_state_.rand(self.n_inputs)
        # self.W_bias = np.arctanh(r0) - np.dot(self.W, r0) - np.dot(self.W_in, x0)

    def func(self, state, input_pattern):
        temp = state * (1 - self.leaky_rate) + self.leaky_rate * \
               np.tanh(np.dot(self.W, state) + np.dot(self.W_in, input_pattern) + self.W_bias)
        return temp

    def fit(self, inputs, outputs=None):
        """
        inputs: np.ndarray
                    shape=[n_input, time_legnth]
        control_par: np.ndarray
                    shape=[time_legnth]
        """
        if self.scaling is not None:
            inputs = np.dot(np.diag(self.scaling), inputs)
        if outputs is None:
            outputs = inputs[:, 1:]

        n_iteration = outputs.shape[1]
        if not self.silent:
            print("harvesting states...")
            print(f"train length {n_iteration}")
        states = np.zeros((self.n_reservoir, n_iteration))
        for n in range(n_iteration):
            self.state = self.func(self.state, inputs[:, n])
            states[:, n] = self.state

        # washout
        self.washout = max(int(n_iteration / 10), self.washout)
        outputs = outputs[:, self.washout:]
        states = states[:, self.washout:]
        yxt = np.dot(outputs, states.T)
        xxt = np.dot(states, states.T)
        self.W_lr = np.dot(yxt, np.linalg.inv(xxt + self.ridge_param * np.eye(self.n_reservoir)))

        pred_train = np.dot(self.W_lr, states)
        pred_train = np.dot(np.diag(1 / self.scaling), pred_train)
        pred_label = np.dot(np.diag(1 / self.scaling), outputs)
        if not self.silent:
            print("mse:", np.mean(np.linalg.norm(pred_train - pred_label, ord=2,  axis=0)))
        return pred_train

    def func_forward(self, state):
        input_pattern = np.dot(self.W_lr, state)
        temp = state * (1 - self.leaky_rate) + self.leaky_rate * \
               np.tanh(np.dot(self.W, state) + np.dot(self.W_in, input_pattern) + self.W_bias)
        return temp

    def forward(self, input, n_iteration):
        input = np.dot(np.diag(self.scaling), input)
        outputs = np.zeros((self.n_outputs, n_iteration))
        self.state = self.func(self.state, input)
        outputs[:, 0] = np.dot(self.W_lr, self.state)
        for n in range(1, n_iteration):
            self.state = self.func_forward(self.state)
            outputs[:, n] = np.dot(self.W_lr, self.state)
        outputs = np.dot(np.diag(1 / self.scaling), outputs)
        return outputs

