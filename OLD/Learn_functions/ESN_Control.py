# -*- coding: utf-8 -*- 
# @Time : 2022/2/16 14:28 
# @Author : lepold
# @File : ESN.py


import numpy as np
from utils import progress_bar
import time

class Esn(object):
    def __init__(self, n_inputs, n_outputs, n_reservoir=200, gamma=1.,
                 sparsity=0., random_state=None, inverse='pinv', spectral_radius=0.95,
                 ridge_param=1e-5, delta_t=0.01, silent=True, **kwargs):

        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.gamma = gamma
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        self.delta_t = delta_t
        self.valid_index = kwargs.get("valid_index") - 1

        self.inverse = inverse
        self.ridge_param = ridge_param
        self.random_state = random_state

        self.state = np.zeros(n_reservoir)
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
        self.init_weights()

    def init_weights(self, sig=0.008, c=0.004):
        W = (self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5) * 2  # [-1, 1]
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / radius)

        self.W_in = (self.random_state_.rand(self.n_reservoir, self.n_inputs) - 0.5) * 2 * sig
        self.W_control = (self.random_state_.rand(self.n_reservoir, 1) - 0.5) * 2 * c

        # fixed point
        r0 = self.random_state_.rand(self.n_reservoir) * 0.2 + 0.8 * np.sign(
            self.random_state_.rand(self.n_reservoir) - 0.5)
        x0 = np.zeros(self.n_inputs)
        c0 = np.zeros(1)
        self.w_bias = np.arctanh(r0) - np.dot(self.W, r0) - np.dot(self.W_in, x0) - np.dot(self.W_control, c0)


    def func(self, state, input_pattern, control_par):
        temp = (-state + np.tanh(np.dot(self.W, state) + np.dot(self.W_in, input_pattern) +
                          np.dot(self.W_control, control_par) + self.w_bias)) * self.gamma
        return temp

    def fit(self, inputs, control_par, outputs=None):
        """
        inputs: np.ndarray
                    shape=[n_input, time_legnth]
        control_par: np.ndarray
                    shape=[time_legnth]
        """
        if outputs is None:
            outputs = inputs[:, 1:, 0]

        n_iteration = outputs.shape[1]
        if not self.silent:
            print("harvesting states...")
            print(f"train length {n_iteration}")
        states = np.zeros((self.n_reservoir, n_iteration))
        computation_start = time.time()
        for n in range(n_iteration):
            progress = n / n_iteration
            progress_bar(progress, time.time() - computation_start)
            k1 = self.delta_t * self.func(self.state, inputs[:, n, 0], control_par[:, n, 0])
            k2 = self.delta_t * self.func(self.state + k1 / 2, inputs[:, n, 1], control_par[:, n, 1])
            k3 = self.delta_t * self.func(self.state + k2 / 2, inputs[:, n, 2], control_par[:, n, 2])
            k4 = self.delta_t * self.func(self.state + k3, inputs[:, n, 3], control_par[:, n, 3])
            self.state = self.state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            states[:, n] = self.state

        # the same as paper

        outputs = outputs[:, self.valid_index]
        states = states[:, self.valid_index]
        # yxt = np.dot(outputs, states.T)
        # xxt = np.dot(states, states.T)
        # self.W_lr = np.dot(yxt, np.linalg.inv(xxt + self.ridge_param * np.eye(self.n_reservoir)))
        temp = np.linalg.lstsq(states.T, outputs.T, rcond=None)[0]
        self.W_lr = temp.T
        # self.state = states[:, int(states.shape[1] / 40)]

        pred_train = np.dot(self.W_lr, states)
        if not self.silent:
            print("mse:", np.sqrt(np.sum((pred_train - outputs) ** 2)))
        return pred_train

    def func_forward(self, state, control_par):
        input_pattern = np.dot(self.W_lr, state)
        temp = (-state + np.tanh(np.dot(self.W, state) + np.dot(self.W_in, input_pattern) +
                          np.dot(self.W_control, control_par) + self.w_bias)) * self.gamma
        return temp

    def forward(self, control_par):
        n_iteration = control_par.shape[1]
        outputs = np.zeros((self.n_outputs, n_iteration))
        computation_start = time.time()
        for n in range(n_iteration):
            # if n>32000:
            #     print("debug here")
            progress = n / n_iteration
            progress_bar(progress, time.time() - computation_start)
            k1 = self.delta_t * self.func_forward(self.state, control_par[:, n, 0])
            k2 = self.delta_t * self.func_forward(self.state + k1 / 2, control_par[:, n, 1])
            k3 = self.delta_t * self.func_forward(self.state + k2 / 2, control_par[:, n, 2])
            k4 = self.delta_t * self.func_forward(self.state + k3, control_par[:, n, 3])
            self.state = self.state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            outputs[:, n] = np.dot(self.W_lr, self.state)
        return outputs

