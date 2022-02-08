# -*- coding: utf-8 -*- 
# @Time : 2022/1/11 17:10 
# @Author : lepold
# @File : RAFDA.py


import numpy as np
import unittest
import matplotlib.pyplot as plt


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

class RandomFeatureMap():

    @staticmethod
    def outer_product_sum(A, B=None):
        r"""
        Computes the sum of the outer products of the rows in A and B
            P = \Sum {A[:, [i]] B[:, [i]].T} for i in 0..N
            Notionally:
            P = 0
            for y in A.T:
                P += np.outer(y, y)
        This is a standard computation for sigma points used in the UKF, ensemble
        Kalman filter, etc., where A would be the residual of the sigma points
        and the filter's state or measurement.
        The computation is vectorized, so it is much faster than the for loop
        for large A.
        Parameters
        ----------
        A : np.array, shape (M, N)
            columns of M-vectors to have the outer product summed
        B : np.array, shape (M, N)
            columns of M-vectors to have the outer product summed
            If it is `None`, it is set to A.
        Returns
        -------
        P : np.array, shape(N, N)
            sum of the outer product of the columns of A and B

        Note: it's also equaling to sum of Khatri-Rao Product in the last dimension.
        """

        if B is None:
            B = A

        outer = np.einsum('ji,ki->jki', A, B)
        return np.sum(outer, axis=-1)

    def __init__(self, W_in, b_in, ridge_param=1e-3):
        assert isinstance(W_in, np.ndarray)
        assert isinstance(b_in, np.ndarray)
        assert W_in.shape[0] == b_in.shape[0]
        self.W_in = W_in
        self.b_in = b_in
        self.D_r = W_in.shape[0]  # reservoir_dim
        self.D = W_in.shape[1]  # input_dim
        self.W_lr_dim = self.D_r * self.D
        self.ridge_param = ridge_param
        self.W_lr = np.zeros((self.D, self.D_r))  # shape=[output_dim, reservoir_dim]

    def fit_lr(self, u_ob, label=None):
        assert u_ob.ndim == 2
        input_dim, seq_len = u_ob.shape
        if label is None:
            label = u_ob[:, 1:]
        seq_len = seq_len - 1
        Phi = []
        for i in range(seq_len):
            phi = np.tanh(self.W_in @ u_ob[:, i] + self.b_in)
            Phi.append(phi)
        Phi = np.stack(Phi, axis=1)  # shape [reservoir_dim, seq_len]
        # now to sove W_lr @ X = Y
        yxt = np.dot(label, Phi.T)
        xxt = np.dot(Phi, Phi.T)
        self.W_lr = np.dot(yxt, np.linalg.inv(xxt + self.ridge_param * np.eye(self.D_r)))

    def forward(self, u_input, time):
        out = [u_input]
        u = u_input.copy()
        for i in range(time - 1):
            u = self.W_lr @ np.tanh(np.dot(self.W_in, u) + self.b_in)
            out.append(u)
        out = np.stack(out, axis=1)
        return out

    def fit_da(self, u_ob, label=None, ensembles=100, eta=0.2, gamma=1000., if_save_state=False):
        self.fit_lr(u_ob, label)
        seq_len = u_ob.shape[1] - 1
        seq_len = seq_len if seq_len < 1000 else 1000
        u_cov = np.eye(self.D) * eta  # observation uncertainty
        W_cov = np.eye(self.W_lr_dim) * gamma

        B = np.zeros((self.W_lr_dim, self.D), dtype=np.float32)
        for i in range(self.D):
            B[i * self.D_r: (i + 1) * self.D_r, i] = 1.

        # initial post distribution
        u_post = (u_ob[:, 0] + np.random.multivariate_normal(np.zeros(self.D), cov=u_cov, size=ensembles)).T
        W_post = (self.W_lr.reshape(-1) + np.random.multivariate_normal(np.zeros(self.W_lr_dim), cov=W_cov,
                                                                        size=ensembles)).T

        if if_save_state:
            total_W = [W_post]
            total_u = [u_post]

        for idx in range(seq_len):
            print(f" da idx {idx}", end="\r")
            # forecast
            W_lr = W_post.reshape((self.D, self.D_r, ensembles))
            temp = np.tanh(np.dot(self.W_in, u_post) + self.b_in[:, np.newaxis])
            u_forecast = np.einsum("jki, ki->ji", W_lr, temp)
            W_forecast = W_post

            # filter
            u_diff = u_forecast - np.mean(u_forecast, axis=1, keepdims=True)
            W_diff = W_forecast - np.mean(W_forecast, axis=1, keepdims=True)
            P_uu = self.outer_product_sum(u_diff) / (ensembles - 1)
            P_wu = self.outer_product_sum(W_diff, u_diff) / (ensembles - 1)
            P_wu = P_wu  # * B # * 1.0002 # localization to mitigate against possible spurious correlations.

            u_ob_noise = u_ob[:,
                         [idx + 1]]  # + np.random.multivariate_normal(np.zeros(self.D), cov=u_cov, size=ensembles).T
            R, _, _, _ = np.linalg.lstsq(P_uu + u_cov, u_forecast - u_ob_noise)
            u_post = u_forecast - P_uu @ R
            W_post = W_forecast - P_wu @ R
            # u_post = u_forecast - P_uu @ np.linalg.inv(P_uu + u_cov) @ (u_forecast - u_ob_noise)
            # W_post = W_forecast - P_wu @ np.linalg.inv(P_uu + u_cov) @ (u_forecast - u_ob_noise)
            if if_save_state:
                total_W.append(W_post)
                total_u.append(u_post)
        self.W_lr = W_post.mean(axis=1).reshape((self.D, self.D_r))
        if if_save_state:
            total_W = np.array(total_W)
            total_u = np.array(total_u)
            np.savez("da_process_state.npz", total_W=total_W, total_u=total_u)


class TestBlock(unittest.TestCase):
    @staticmethod
    def loss(x, y):
        loss = np.linalg.norm(x - y, ord=2, axis=0)
        return loss

    def _test_lorenz(self):
        lorenz = Lorenz63(10., 28., 8 / 3, delta_t=0.02)
        lorenz.x, lorenz.y, lorenz.z = 1., 2., -1.
        states = []
        for i in range(4040):
            state = lorenz.run()
            states.append(state)
        states = np.stack(states, axis=1)
        states = states[:, 40:]
        # np.save("lorenz_attractor.npy", states)

        # Plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot(states[0], states[1], states[2], lw=0.5)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")
        plt.show()

    def _test_run_lr(self):
        # u = np.load("lorenz_attractor.npy")
        # u_test = np.load("lorenz_attractor2.npy")
        lorenz = Lorenz63(10., 28., 8 / 3, delta_t=0.02)
        u = lorenz.res(20100, 2)
        u_test = lorenz.res(5000, 3)

        dt = 0.02
        D_r = 300
        N = 2000
        eta = 0.2

        ridge_param = 4e-5
        D = 3
        lyapunov_exp = 0.91
        W_in = np.random.uniform(-0.005, 0.005, size=(D_r, D))
        b = np.random.uniform(-4, 4, size=(D_r))
        rfda = RandomFeatureMap(W_in, b, ridge_param)
        rfda.fit_lr(u)
        time = 500
        max_time_unit = int(time * dt / lyapunov_exp)
        time_ticks = [l / lyapunov_exp / dt for l in range(max_time_unit)]
        u_predict = rfda.forward(u_test[:, 0], time=time)
        fig, ax = plt.subplots(4, 1, figsize=(8, 6))
        ax = ax.flatten()
        coords = ["x coordinate", "y coordinate", "z coordinate"]
        loss = self.loss(u_test[:, :time], u_predict)
        for i in range(3):
            ax[i].plot(range(time), u_test[i, :time], 'k', lw=1, label="target system")
            ax[i].plot(range(time), u_predict[i, :], 'r', lw=1, label="free running RFDA")
            ax[i].set_xticks([])
            ax[i].text(0.1, 0.8, coords[i], fontsize=10, ha='center', va='center', color='b', transform=ax[i].transAxes)
        ax[3].plot(range(time), loss, 'k', lw=1, label="loss")
        ax[3].legend(loc=(0.05, 0.7), fontsize=12)
        ax[0].legend(loc=(0.61, 1.1), fontsize='x-small')
        ax[3].set_xticks(time_ticks)
        ax[3].set_xticklabels(np.arange(max_time_unit))
        ax[3].set_xlabel('$ \lambda_{max}t $')
        ax[0].set_title("LR_without noise")

        plt.savefig("./LR_predict_lorenz.png")
        plt.show()

    def _test_run_lr_noise(self):
        # u = np.load("lorenz_attractor.npy")
        # u_test = np.load("lorenz_attractor2.npy")
        lorenz = Lorenz63(10., 28., 8 / 3, delta_t=0.02)
        u = lorenz.res(20100, 2)
        u_test = lorenz.res(5000, 3)
        dt = 0.02
        D_r = 300
        N = 2000
        eta = 0.2
        D, seq_len = u.shape
        u_noise = u + np.random.multivariate_normal(np.zeros(D), cov=np.eye(D) * eta, size=seq_len).T

        ridge_param = 4e-5
        D = 3
        lyapunov_exp = 0.91
        W_in = np.random.uniform(-0.005, 0.005, size=(D_r, D))
        b = np.random.uniform(-4, 4, size=(D_r))
        rfda = RandomFeatureMap(W_in, b, ridge_param)
        rfda.fit_lr(u_noise)
        time = 500
        max_time_unit = int(time * dt / lyapunov_exp)
        time_ticks = [l / lyapunov_exp / dt for l in range(max_time_unit)]
        u_predict = rfda.forward(u_test[:, 0], time=time)
        fig, ax = plt.subplots(4, 1, figsize=(8, 6))
        ax = ax.flatten()
        coords = ["x coordinate", "y coordinate", "z coordinate"]
        loss = self.loss(u_test[:, :time], u_predict)
        for i in range(3):
            ax[i].plot(range(time), u_test[i, :time], 'k', lw=1, label="target system")
            ax[i].plot(range(time), u_predict[i, :], 'r', lw=1, label="free running RFDA")
            ax[i].set_xticks([])
            ax[i].text(0.1, 0.8, coords[i], fontsize=10, ha='center', va='center', color='b', transform=ax[i].transAxes)
        ax[3].plot(range(time), loss, 'k', lw=1, label="loss")
        ax[3].legend(loc=(0.05, 0.7), fontsize=12)
        ax[0].legend(loc=(0.61, 1.1), fontsize='x-small')
        ax[3].set_xticks(time_ticks)
        ax[3].set_xticklabels(np.arange(max_time_unit))
        ax[3].set_xlabel('$ \lambda_{max}t $')
        ax[0].set_title("LR_with noise")
        plt.savefig("./LR_predict_noiselorenz.png")
        plt.show()

    def test_run_da_noise(self):
        # u = np.load("lorenz_attractor.npy")
        # u_test = np.load("lorenz_attractor2.npy")
        lorenz = Lorenz63(10., 28., 8 / 3, delta_t=0.02)
        u = lorenz.res(4040, 2)
        u_test = lorenz.res(4040, 3)

        dt = 0.02
        D_r = 300
        eta = 0.2
        D, seq_len = u.shape
        print(f"\nD {D}, train seq_len {seq_len}")
        u_noise = u + np.random.multivariate_normal(np.zeros(D), cov=np.eye(D) * eta, size=seq_len).T

        ridge_param = 4e-5
        D = 3
        lyapunov_exp = 0.91
        W_in = np.random.uniform(-0.005, 0.005, size=(D_r, D))
        b = np.random.uniform(-4, 4, size=(D_r))
        rfda = RandomFeatureMap(W_in, b, ridge_param)
        rfda.fit_da(u_noise, None, ensembles=300, eta=eta, gamma=1000, if_save_state=False)
        time = 500
        max_time_unit = int(time * dt / lyapunov_exp)
        time_ticks = [l / lyapunov_exp / dt for l in range(max_time_unit)]
        u_predict = rfda.forward(u_test[:, 0], time=time)

        fig, ax = plt.subplots(4, 1, figsize=(8, 6))
        ax = ax.flatten()
        coords = ["x coordinate", "y coordinate", "z coordinate"]
        loss = self.loss(u_test[:, :time], u_predict)
        for i in range(3):
            ax[i].plot(range(time), u_test[i, :time], 'k', lw=1, label="target system")
            ax[i].plot(range(time), u_predict[i, :], 'r', lw=1, label="free running RFDA")
            ax[i].set_xticks([])
            ax[i].text(0.1, 0.8, coords[i], fontsize=10, ha='center', va='center', color='b', transform=ax[i].transAxes)
        ax[3].plot(range(time), loss, 'k', lw=1, label="loss")
        ax[3].legend(loc=(0.05, 0.7), fontsize=12)
        ax[0].legend(loc=(0.61, 1.1), fontsize='x-small')
        ax[3].set_xticks(time_ticks)
        ax[3].set_xticklabels(np.arange(max_time_unit))
        ax[3].set_xlabel('$ \lambda_{max}t $')

        ax[0].set_title("DA_with noise")
        plt.savefig("./DA_predict_noiselorenz.png")
        plt.show()


if __name__ == '__main__':
    unittest.main()
