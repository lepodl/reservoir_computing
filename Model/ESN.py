# -*- coding: utf-8 -*- 
# @Time : 2022/1/25 20:19 
# @Author : lepold
# @File : Esn.py

import numpy as np


def correct_dimensions(s, targetlength):
    """checks the dimensionality of some numeric argument s, broadcasts it
       to the specified length if possible.
    Args:
        s: None, scalar or 1D array
        targetlength: expected length of s
    Returns:
        None if s is None, else numpy vector of length targetlength
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s


def identity(x):
    return x


class ESN():

    def __init__(self, n_inputs, n_outputs, n_reservoir=200, leaky_rate=1.,
                 spectral_radius=0.95, sparsity=0., noise=0.001,
                 input_scaling=None, teacher_scaling=None, extended_states=False,
                 out_activation=identity, random_state=None, inverse='pinv',
                 ridge_param=1e-5, silent=True):
        """
        Args:
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            n_reservoir: nr of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
            noise: noise added to each neuron (regularization)

            input_scaling: scalar or vector of length n_inputs to multiply
                        with each input dimension before feeding it to the netw.
            teacher_scaling: factor applied to the target signal
            out_activation: output activation function (applied to the readout)
            random_state: positive integer seed, np.rand.RandomState object,
                          or None to use numpy's builting RandomState.
            silent: supress messages
        """
        # check for proper dimensionality of all arguments and write them down.

        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.leaky_rate = leaky_rate
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)
        self.teacher_scaling = correct_dimensions(teacher_scaling, n_outputs)

        self.inverse = inverse
        self.ridge_param = ridge_param
        self.extended_states = extended_states

        self.out_activation = out_activation
        self.random_state = random_state

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
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
        self.initweights()

    def initweights(self):
        # initialize recurrent weights:
        # begin with a random matrix centered around zero:
        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # delete the fraction of connections given by (self.sparsity):
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        # rescale them to reach the requested spectral radius:
        self.W = W * (self.spectral_radius / radius)

        # random input weights:
        self.W_in = self.random_state_.rand(
            self.n_reservoir, self.n_inputs) * 2 - 1
        self.w_bias = self.random_state_.rand(self.n_reservoir) * 2 - 1

    def _update(self, state, input_pattern):
        """performs one update step.
        i.e., computes the next network state by applying the recurrent weights
        to the last state.
        """
        preactivation = (np.dot(self.W, state)
                         + np.dot(self.W_in, input_pattern) + self.w_bias)

        out = self.leaky_rate * (
                np.tanh(preactivation) + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5)) + (
                      1 - self.leaky_rate) * state
        return out

    def _scale_inputs(self, inputs):
        """for each input dimension j: multiplies by the j'th entry in the
        input_scaling argument, then adds the j'th entry of the input_shift
        argument."""
        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        return inputs

    def _scale_teacher(self, teacher):
        """multiplies the teacher/target signal by the teacher_scaling argument,
        then adds the teacher_shift argument to it."""
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        return teacher

    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    def fit(self, inputs, outputs, inspect=False):
        """
        Collect the network's reaction to training data, train readout weights.
        Args:
            inputs: array of dimensions (N_training_samples x n_inputs)
            outputs: array of dimension (N_training_samples x n_outputs)
            inspect: show a visualisation of the collected reservoir states
        Returns:
            the network's output on the training data, using the trained weights
        """
        # transform any vectors of shape (x,) into vectors of shape (x,1):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        # transform input and teacher signal:
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)

        if not self.silent:
            print("harvesting states...")
        # step the reservoir through the given input,output pairs:
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._update(states[n - 1], inputs_scaled[n, :])

        # learn the weights, i.e. find the linear combination of collected
        # network states that is closest to the target output
        if not self.silent:
            print("fitting...")
        # we'll disregard the first few states:
        transient = min(int(inputs.shape[0] / 10), 100)
        # include the raw inputs:
        if self.extended_states:
            extended_states = np.hstack((states, inputs_scaled))
        else:
            extended_states = states
        ll = extended_states.shape[1]
        # Solve for W_out:
        if self.inverse == "pinv":
            self.W_out = np.dot(np.linalg.pinv(extended_states[transient:, :]),
                                teachers_scaled[transient:, :])  # shape: [extended_dim, output_dim]
        elif self.inverse == "cholesky":
            XTY = extended_states[transient:, :].T @ teachers_scaled[transient:, :]
            XTX = extended_states[transient:, :].T @ extended_states[transient:, :] + self.ridge_param * np.eye(ll)
            self.W_out, _, _, _ = np.linalg.lstsq(XTX, XTY)

        # remember the last state for later:
        self.laststate = states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = teachers_scaled[-1, :]

        # optionally visualize the collected states
        if inspect:
            from matplotlib import pyplot as plt
            # (^-- we depend on matplotlib only if this option is used)
            plt.figure(
                figsize=(states.shape[0] * 0.0025, states.shape[1] * 0.01))
            plt.imshow(extended_states.T, aspect='auto',
                       interpolation='nearest')
            plt.colorbar()

        if not self.silent:
            print("training error:")
        # apply learned weights to the collected states:
        pred_train = self._unscale_teacher(
            np.dot(extended_states, self.W_out))
        if not self.silent:
            print("mse:", np.sqrt(np.mean((pred_train - outputs) ** 2)))
        return pred_train

    def da(self, inputs, outputs, Enkf, random_walk=0.01, observation_noise=0.2):
        if self.extended_states:
            raise NotImplementedError
        self.fit(inputs, outputs, inspect=False)
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        # transform input and teacher signal:
        inputs_scaled = self._scale_inputs(inputs)
        if not self.silent:
            print("DA states...")
        initial_states = np.concatenate((self.W_out.flatten(), inputs_scaled[0]))
        self.Wout_dims = self.n_reservoir * self.n_outputs
        P = np.eye(initial_states.shape[0]) * random_walk * 3
        P[self.Wout_dims:, self.Wout_dims:] = np.eye(
            self.n_inputs) * observation_noise * self.teacher_scaling  # can be generated through np.diag()
        Q_noise = 0.
        R_noise = np.eye(self.n_outputs) * observation_noise * self.teacher_scaling
        zero_mean = np.zeros(self.Wout_dims)
        random_cov = np.eye(self.Wout_dims) * random_walk
        self.last_states = np.tile(self.laststate, (100, 1)).T

        def _fx(x, dt=None, ):
            # x: [w_out, u]
            x_copy = x.copy()
            preactivation = (np.dot(self.W, self.last_states)
                             + np.dot(self.W_in, x[self.Wout_dims:]) + self.w_bias[:, np.newaxis])

            out = self.leaky_rate * (
                    np.tanh(preactivation) + self.noise * (self.random_state_.rand(self.n_reservoir, 1) - 0.5)) + (
                          1 - self.leaky_rate) * self.last_states

            self.last_states = out
            W_out = x[:self.Wout_dims] + np.random.multivariate_normal(zero_mean, cov=random_cov, size=100).T
            W_outs = W_out.reshape((100, self.n_outputs, self.n_reservoir))
            x_copy[self.Wout_dims:] = np.einsum('ijk, ki->ji', W_outs, out)
            x_copy[:self.Wout_dims] = W_out
            return x_copy

        def _hx(x):
            return x[self.Wout_dims:]

        enkf = Enkf(initial_states, P, self.n_outputs, None, 100, _hx, _fx, Q_noise, R_noise)
        for i in range(1, inputs_scaled.shape[0]):
            print(f"DA {i}", end="\r")
            enkf.predict()
            enkf.update(inputs_scaled[i])
        print("enkf.xpost.shape", enkf.x_post.shape)
        self.W_out = enkf.x_post[:self.Wout_dims].squeeze().reshape((self.n_outputs, self.n_reservoir))

    def predict(self, input, seq_len=1000, reset_state=False):
        """
        Apply the learned weights to the network's reactions to new input.
        Args:
            input: array of dimensions (n_inputs, 1) or (n_inputs)
            seq_len: time length to predict
            reset_state: if False, start the network from the last training state
        Returns:
            Array of output activations
        """

        assert isinstance(input, np.ndarray)
        if input.ndim == 1:
            input = input[np.newaxis, :]
        input = self._scale_inputs(input)

        if not reset_state:
            laststate = self.laststate
            # lastinput = self.lastinput
            # lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_reservoir)
            # lastinput = np.zeros(self.n_inputs)
            # lastoutput = np.zeros(self.n_outputs)

        outputs = np.zeros((seq_len, self.n_outputs))
        outputs = np.vstack((input, outputs))
        if self.extended_states:
            for n in range(seq_len):
                laststate = self._update(laststate, outputs[n])
                outputs[n + 1, :] = np.dot(np.concatenate([laststate, outputs[n, :]]), self.W_out)

        else:
            for n in range(seq_len):
                laststate = self._update(laststate, outputs[n, :])
                outputs[n + 1, :] = np.dot(laststate, self.W_out)

        return self._unscale_teacher(outputs[1:])
