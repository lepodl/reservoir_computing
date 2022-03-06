# -*- coding: utf-8 -*- 
# @Time : 2022/2/28 16:44 
# @Author : lepold
# @File : Esn.py

import numpy as np


class Esn(object):
    @staticmethod
    def outer_product_sum(A, B=None):
        if B is None:
            B = A
        outer = np.einsum('ji,ki->jki', A, B)
        return np.sum(outer, axis=-1)

    def __init__(self, n_inputs, n_outputs, n_reservoir=200, leaky_rate=0.1,
                 sparsity=0., random_state=None, spectral_radius=0.95,
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
        self.init_weights(sig=0.008)

    def init_weights(self, sig=0.008):
        W = (self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5) * 2  # [-1, 1]
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / radius)

        self.W_in = (self.random_state_.rand(self.n_reservoir, self.n_inputs) - 0.5) * 2 * sig

        # fixed point
        r0 = self.random_state_.rand(self.n_reservoir) * 0.2 + 0.8 * np.sign(
            self.random_state_.rand(self.n_reservoir) - 0.5)
        x0 = self.random_state_.rand(self.n_inputs)
        self.W_bias = np.arctanh(r0) - np.dot(self.W, r0) - np.dot(self.W_in, x0)

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

        # for plot total series, add the last point of input
        temp = self.func(self.state, inputs[:, -1])

        outputs = outputs[:, self.washout:]
        states = states[:, self.washout:]
        yxt = np.dot(outputs, states.T)
        xxt = np.dot(states, states.T)
        # tempp = np.linalg.lstsq(states.T, outputs.T, rcond=None)[0]
        # self.W_lr = tempp.T
        self.W_lr = np.dot(yxt, np.linalg.inv(xxt + self.ridge_param * np.eye(self.n_reservoir)))

        states = np.concatenate([states, temp[:, np.newaxis]], axis=1)
        pred_train = np.dot(self.W_lr, states)
        if not self.silent:
            print("mse:", np.mean(np.sqrt(np.sum((pred_train[:, :-1] - outputs) ** 2, axis=0))))
        return pred_train

    def fit_da(self, inputs, outputs=None, ensembles=100, eta=0.1, gamma=1000., initial_zero=False,
               return_train_prediction=False, return_total_Wlr=False):

        n_iteration = inputs.shape[1] - 1
        # n_iteration = n_iteration if n_iteration < 1000 else 1000

        u_cov = np.eye(self.n_inputs) * eta  # observation uncertainty
        W_cov = np.eye(self.W_lr_dim) * gamma

        B = np.zeros((self.W_lr_dim, self.n_outputs), dtype=np.float32)
        for i in range(self.n_outputs):
            B[i * self.n_reservoir: (i + 1) * self.n_reservoir, i] = 1.

        # initial post distribution
        u_post = (inputs[:, 0] + np.random.multivariate_normal(np.zeros(self.n_inputs), cov=u_cov, size=ensembles)).T
        if not initial_zero:
            W_post = (self.W_lr.reshape(-1) + np.random.multivariate_normal(np.zeros(self.W_lr_dim), cov=W_cov,
                                                                            size=ensembles)).T
            self.fit(inputs, outputs)
            states = np.broadcast_to(self.state[:, np.newaxis], (self.n_reservoir, ensembles))
        else:
            W_post = np.random.multivariate_normal(np.zeros(self.W_lr_dim), cov=W_cov, size=ensembles).T
            # self.fit(inputs[:, :self.washout])
            states = np.broadcast_to(self.state[:, np.newaxis], (self.n_reservoir, ensembles))

        if return_total_Wlr:
            total_W = [self.W_lr]
        else:
            total_W = None

        for idx in range(n_iteration):
            # print(f" da idx {idx}", end="\r")
            # forecast
            W_lr = W_post.reshape((self.n_outputs, self.n_reservoir, ensembles))
            states = states * (1 - self.leaky_rate) + self.leaky_rate * np.tanh(np.einsum('jk, ki->ji', self.W, states) \
                        + np.einsum('jk, ki->ji', self.W_in, u_post) + self.W_bias[:, np.newaxis])
            u_forecast = np.einsum("jki, ki->ji", W_lr, states)
            W_forecast = W_post

            # filter
            u_diff = u_forecast - np.mean(u_forecast, axis=1, keepdims=True)
            W_diff = W_forecast - np.mean(W_forecast, axis=1, keepdims=True)
            P_uu = self.outer_product_sum(u_diff) / (ensembles - 1)
            P_wu = self.outer_product_sum(W_diff, u_diff) / (ensembles - 1)
            P_wu = P_wu  # * B * 1.0002 # localization to mitigate against possible spurious correlations.

            u_ob_noise = inputs[:,
                         [
                             idx + 1]] + np.random.multivariate_normal(np.zeros(self.n_inputs), cov=u_cov, size=ensembles).T
            R, _, _, _ = np.linalg.lstsq(P_uu + u_cov, u_forecast - u_ob_noise, rcond=None)
            u_post = u_forecast - P_uu @ R
            W_post = W_forecast - P_wu @ R
            # u_post = u_forecast - P_uu @ np.linalg.inv(P_uu + u_cov) @ (u_forecast - u_ob_noise)
            # W_post = W_forecast - P_wu @ np.linalg.inv(P_uu + u_cov) @ (u_forecast - u_ob_noise)
            self.W_lr = W_post.mean(axis=1).reshape((self.n_outputs, self.n_reservoir))
            if idx % int((n_iteration - 1)//4) == 1 and return_total_Wlr:
                total_W.append(self.W_lr)

        train_prediction = None
        if not return_train_prediction:
            state = np.zeros(self.n_reservoir)
            total_states = np.zeros((self.n_reservoir, inputs.shape[1]))
            for i in range(inputs.shape[1]):
                state = self.func(state, inputs[:, i])
                total_states[:, i] = state
            train_prediction = np.dot(self.W_lr, total_states[:, self.washout:])
        return train_prediction, np.array(total_W)

    def func_forward(self, state):
        input_pattern = np.dot(self.W_lr, state)
        temp = state * (1 - self.leaky_rate) + self.leaky_rate * \
               np.tanh(np.dot(self.W, state) + np.dot(self.W_in, input_pattern) + self.W_bias)
        return temp

    def forward(self, input, n_iteration):
        outputs = np.zeros((self.n_outputs, n_iteration))
        self.state = self.func(self.state, input)
        outputs[:, 0] = np.dot(self.W_lr, self.state)
        for n in range(1, n_iteration):
            self.state = self.func_forward(self.state)
            outputs[:, n] = np.dot(self.W_lr, self.state)
        return outputs
