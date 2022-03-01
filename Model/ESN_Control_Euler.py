# -*- coding: utf-8 -*-
# @Time : 2022/2/22 14:28
# @Author : lepold
# @File : ESN.py


import numpy as np
from utils.utils import progress_bar
import time

class Esn(object):
    @staticmethod
    def outer_product_sum(A, B=None):
        if B is None:
            B = A
        outer = np.einsum('ji,ki->jki', A, B)
        return np.sum(outer, axis=-1)

    def __init__(self, n_inputs, n_outputs, n_reservoir=200, leaky_rate=0.98,
                 sparsity=0., random_state=None, spectral_radius=0.95,
                 ridge_param=1e-5, delta_t=0.01, washout=100, silent=True, **kwargs):

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
        self.valid_index = kwargs.get("valid_index", None)

        self.state = np.zeros(n_reservoir)
        self.W = None
        self.W_in = None
        self.W_bias = None
        self.W_lr = np.zeros((n_outputs, n_reservoir))
        self.W_lr_dim = n_outputs * n_reservoir

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
        self.init_weights(sig=0.008, c=0.004)

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
        self.W_bias = np.arctanh(r0) - np.dot(self.W, r0) - np.dot(self.W_in, x0) - np.dot(self.W_control, c0)


    def func(self, state, input_pattern, control_par):
        temp = state * (1 - self.leaky_rate) + self.leaky_rate * \
               np.tanh(np.dot(self.W, state) + np.dot(self.W_in, input_pattern) + np.dot(self.W_control,
                                                                                         control_par) + self.W_bias)
        return temp

    def fit(self, inputs, control_par, outputs=None):
        """
        inputs: np.ndarray
                    shape=[n_input, time_legnth]
        control_par: np.ndarray
                    shape=[time_legnth]
        """
        if outputs is None:
            outputs = inputs

        n_iteration = outputs.shape[1]
        if not self.silent:
            print("harvesting states...")
            print(f"train length {n_iteration}")
        states = np.zeros((self.n_reservoir, n_iteration))
        computation_start = time.time()
        for n in range(n_iteration-1):
            progress = n / n_iteration
            progress_bar(progress, time.time() - computation_start)
            self.state = self.func(self.state, inputs[:, n], control_par[:, n])
            states[:, n+1] = self.state
        if self.valid_index is not None:
            outputs = outputs[:, self.valid_index]
            states = states[:, self.valid_index]
        else:
            outputs = outputs[:, self.washout:]
            states = states[:, self.washout:]
        # yxt = np.dot(outputs, states.T)
        # xxt = np.dot(states, states.T)
        # self.W_lr = np.dot(yxt, np.linalg.inv(xxt + self.ridge_param * np.eye(self.n_reservoir)))
        temp = np.linalg.lstsq(states.T, outputs.T, rcond=None)[0]
        self.W_lr = temp.T
        self.state = states[:, int(states.shape[1] / 40)]

        pred_train = np.dot(self.W_lr, states)
        if not self.silent:
            print("mse:", np.mean(np.sqrt(np.sum((pred_train[:, :-1] - outputs) ** 2, axis=0))))
        return pred_train

    def fit_da(self, inputs, control_par, outputs=None, ensembles=100, eta=0.1, gamma=1000., initial_zero=False,
               return_train_prediction=False):

        n_iteration = inputs.shape[1]
        # n_iteration = n_iteration if n_iteration < 1000 else 1000

        u_cov = np.eye(self.n_inputs) * eta  # observation uncertainty
        W_cov = np.eye(self.W_lr_dim) * gamma

        B = np.zeros((self.W_lr_dim, self.n_reservoir), dtype=np.float32)
        for i in range(self.n_reservoir):
            B[i * self.n_reservoir: (i + 1) * self.n_reservoir, i] = 1.

        # initial post distribution

        if not initial_zero:
            W_post = (self.W_lr.reshape(-1) + np.random.multivariate_normal(np.zeros(self.W_lr_dim), cov=W_cov,
                                                                            size=ensembles)).T
            self.fit(inputs, control_par, outputs)

        else:
            W_post = np.random.multivariate_normal(np.zeros(self.W_lr_dim), cov=W_cov, size=ensembles).T
            # self.fit(inputs[:, :self.washout])
            states = np.broadcast_to(self.state[:, np.newaxis], (self.n_reservoir, ensembles))

        computation_start = time.time()
        print("\n da")
        n = int(n_iteration / 4)
        n_waste = 2000
        n_valid = n - n_waste
        print("n", n)
        for period in range(4):
            for idxx in range(n_waste):
                idxx = idxx + period * n
                self.state = self.func(self.state, inputs[:, idxx], control_par[:, idxx])
            u_temp = np.dot(self.W_lr, self.state)
            u_post = (u_temp + np.random.multivariate_normal(np.zeros(self.n_inputs), cov=u_cov,
                                                                   size=ensembles)).T
            states = np.broadcast_to(self.state[:, np.newaxis], (self.n_reservoir, ensembles))
            for idx in range(n_valid):
                idx = idx + period * n + n_waste
                progress = idx / n_iteration
                progress_bar(progress, time.time() - computation_start)
                # print(f" da idx {idx}", end="\r")
                # forecast
                W_lr = W_post.reshape((self.n_outputs, self.n_reservoir, ensembles))
                states = states * (1 - self.leaky_rate) + self.leaky_rate * np.tanh(np.einsum('jk, ki->ji', self.W, states) \
                                                                                    + np.einsum('jk, ki->ji', self.W_in,
                                                                                                u_post) + np.dot(self.W_control, control_par[:, idx])[:, np.newaxis] + self.W_bias[:,
                                                                                                          np.newaxis])
                self.state = states[:, 0]
                u_forecast = np.einsum("jki, ki->ji", W_lr, states)
                W_forecast = W_post

                # filter
                u_diff = u_forecast - np.mean(u_forecast, axis=1, keepdims=True)
                W_diff = W_forecast - np.mean(W_forecast, axis=1, keepdims=True)
                P_uu = self.outer_product_sum(u_diff) / (ensembles - 1)
                P_wu = self.outer_product_sum(W_diff, u_diff) / (ensembles - 1)
                P_wu = P_wu  # * B # * 1.0002 # localization to mitigate against possible spurious correlations.
                try:
                    u_ob_noise = inputs[:,[idx + 1]]# + np.random.multivariate_normal(np.zeros(self.n_input), cov=u_cov, size=ensembles).T
                except:
                    break
                R, _, _, _ = np.linalg.lstsq(P_uu + u_cov, u_forecast - u_ob_noise, rcond=None)
                u_post = u_forecast - P_uu @ R
                W_post = W_forecast - P_wu @ R
                # u_post = u_forecast - P_uu @ np.linalg.inv(P_uu + u_cov) @ (u_forecast - u_ob_noise)
                # W_post = W_forecast - P_wu @ np.linalg.inv(P_uu + u_cov) @ (u_forecast - u_ob_noise)
            self.W_lr = W_post.mean(axis=1).reshape((self.n_outputs, self.n_reservoir))
        # train_prediction = None
        # if not return_train_prediction:
        #     state = np.zeros(self.n_reservoir)
        #     total_states = np.zeros((self.n_reservoir, inputs.shape[1]))
        #     for i in range(inputs.shape[1]):
        #         state = self.func(state, inputs[:, i])
        #         total_states[:, i] = state
        #     train_prediction = np.dot(self.W_lr, total_states[:, self.washout:])
        # return train_prediction

    def func_forward(self, state, control_par):
        input_pattern = np.dot(self.W_lr, state)
        temp = state * (1 - self.leaky_rate) + self.leaky_rate * \
               np.tanh(np.dot(self.W, state) + np.dot(self.W_in, input_pattern) + np.dot(self.W_control,
                                                                                         control_par) + self.W_bias)
        return temp

    def forward(self, control_par):
        n_iteration = control_par.shape[1]
        outputs = np.zeros((self.n_outputs, n_iteration))
        for n in range(n_iteration):
            self.state = self.func_forward(self.state, control_par[:, n])
            outputs[:, n] = np.dot(self.W_lr, self.state)
        return outputs
