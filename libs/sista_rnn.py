import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.framework.python.ops.arg_scope import add_arg_scope
import numpy as np
from collections import OrderedDict

# original tf.matmul is without an arg_scope decorator
@add_arg_scope
def matmul(*args, **kwargs):
    return tf.matmul(*args, **kwargs)

def dim_reduction(x, w):
    _shape = x.get_shape().as_list()
    x = tf.reshape(x, [-1, _shape[-1]])
    with arg_scope([matmul], transpose_b=True):
        x = matmul(x, w)
    x = tf.reshape(x, _shape[:-1] + [w.get_shape().as_list()[0]])
    return x

class sista_rnn(object):
    def __init__(self, pre_input, now_input, n_hidden, K, gama, lambda1, lambda2, A_initializer, gw_initializer=None):
        '''
        :param pre_input: input tensor with shape [1, batch_size, n_input]
        :param now_input: input tensor with shape [time_steps, batch_size, n_input]
        :param n_hidden: length of hidden state in RNN models
        :param K: length of iterations in the ISTA algorithm
        :param gama: learning rate in the ISTA algorithm
        :param lambda1: weight of the sparse term
        :param lambda2: weight of the coherence term
        :param A_initializer: A initializer

        '''
        self.pre_input = pre_input
        self.now_input = now_input
        input_shape = now_input.get_shape().as_list()
        self.time_steps, self.batch_size, self.n_input = input_shape
        self.n_hidden = n_hidden
        self.K = K
        self.gama = gama
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.A_initializer = A_initializer
        self.gw_initializer = gw_initializer

    def __get_variable(self, name, shape, initializer=tf.contrib.layers.xavier_initializer()):
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=initializer)

    def __soft(self, x, b):
        return tf.sign(x) * tf.nn.relu(tf.abs(x) - b)

    def forward(self):
        '''
        :return: the hidden state with shape: [time_step, batch_size, K * n_hidden],
                 parameters: A dict with A, gama, lambda1, lambda2, h_0
        '''
        # define variables and constants in the sista_rnn model
        with tf.variable_scope('sista_rnn_trainable_variables'):
            A = self.__get_variable('A', None, self.A_initializer)
            gama = self.__get_variable('gama', None, tf.constant(self.gama, tf.float32))
            lambda1 = self.__get_variable('lambda1', None, tf.constant(self.lambda1, tf.float32))
            lambda2 = self.__get_variable('lambda2', None, tf.constant(self.lambda2, tf.float32))
            h_0 = self.__get_variable('h_0', [self.batch_size, self.n_hidden], tf.zeros_initializer)
        parameters = OrderedDict([('A', A),
                                  ('gama', gama),
                                  ('lambda1', lambda1),
                                  ('lambda2', lambda2),
                                  ('h_0', h_0)])
        epsilon = tf.constant(100, dtype=tf.float32)
        At = tf.matrix_transpose(A)
        AtA = tf.matmul(At, A)
        I = tf.eye(self.n_hidden, self.n_hidden)
        one = tf.constant(1.0, dtype=tf.float32)

        # initialize V
        V = one / gama * At

        # initialize W_1
        W_1 = I - lambda2 / gama * AtA
        h_0_all_layers = tf.tile(h_0, [1, self.K])
        h_t_1_all_layers = h_0_all_layers

        # padding for the first time input
        if self.pre_input is None:
            self.input = tf.concat([tf.zeros([1, self.batch_size, self.n_input], tf.float32), self.now_input], axis=0)
        else:
            self.input = tf.concat([self.pre_input, self.now_input], axis=0)

        # sequential ISTA mapping to RNN
        with arg_scope([matmul], transpose_b=True):
            h = []
            for t in range(self.time_steps):
                h_t_1_last_layer = h_t_1_all_layers[:, (self.K - 1) * self.n_hidden:]
                delta = tf.tile(tf.exp(-tf.norm(self.input[t] - self.input[t + 1], axis=1, keep_dims=True)**2 / epsilon), [1, self.n_hidden])
                h_t_kth_layer = self.__soft(matmul(h_t_1_last_layer, W_1) + matmul(self.input[t + 1], V), lambda1 / gama)
                h_t_all_layers = h_t_kth_layer

                for k in range(1, self.K):
                    W_k = lambda2 / gama * I
                    U_k = I - one / gama * AtA
                    h_t_kth_layer = self.__soft(tf.matmul(h_t_1_last_layer, W_k) * delta
                                         + matmul(h_t_kth_layer, U_k) \
                                         - matmul(h_t_kth_layer, W_k) * delta
                                         + matmul(self.input[t + 1], V), lambda1 / gama)
                    h_t_all_layers = tf.concat([h_t_all_layers, h_t_kth_layer], axis=1)
                h.append(h_t_all_layers)
                h_t_1_all_layers = h_t_all_layers
            h = tf.stack(h)

        return h, parameters

    def forward_As(self):
        '''
        :return: the hidden state with shape: [time_step, batch_size, K * n_hidden],
                 parameters: A dict with A, gama, lambda1, lambda2, h_0
        '''
        # define variables and constants in the sista_rnn model
        As = []
        with tf.variable_scope('sista_rnn_trainable_variables'):
            for i in range(self.K):
                As.append(self.__get_variable('A_{}'.format(i + 1), None, self.A_initializer))
            gama = self.__get_variable('gama', None, tf.constant(self.gama, tf.float32))
            lambda1 = self.__get_variable('lambda1', None, tf.constant(self.lambda1, tf.float32))
            lambda2 = self.__get_variable('lambda2', None, tf.constant(self.lambda2, tf.float32))
            h_0 = self.__get_variable('h_0', [self.batch_size, self.n_hidden], tf.zeros_initializer)
        parameters = OrderedDict([('A_{}'.format(i + 1), As[i]) for i in range(self.K)] + [
                                  ('gama', gama),
                                  ('lambda1', lambda1),
                                  ('lambda2', lambda2),
                                  ('h_0', h_0)])
        epsilon = tf.constant(100, dtype=tf.float32)
        Ats = [tf.matrix_transpose(As[i]) for i in range(self.K)]
        AtAs = [tf.matmul(Ats[i], As[i]) for i in range(self.K)]
        I = tf.eye(self.n_hidden, self.n_hidden)
        one = tf.constant(1.0, dtype=tf.float32)

        # initialize V
        V = one / gama * Ats[0]

        # initialize W_1
        W_1 = I - lambda2 / gama * AtAs[0]

        h_0_all_layers = tf.tile(h_0, [1, self.K])
        h_t_1_all_layers = h_0_all_layers

        # padding for the first time input
        if self.pre_input is None:
            self.input = tf.concat([tf.zeros([1, self.batch_size, self.n_input], tf.float32), self.now_input], axis=0)
        else:
            self.input = tf.concat([self.pre_input, self.now_input], axis=0)
        # sequential ISTA mapping to RNN
        with arg_scope([matmul], transpose_b=True):
            h = []
            for t in range(self.time_steps):
                h_t_1_last_layer = h_t_1_all_layers[:, (self.K - 1) * self.n_hidden:]
                delta = tf.tile(tf.exp(-tf.norm(self.input[t] - self.input[t + 1], axis=1, keep_dims=True)**2 / epsilon), [1, self.n_hidden])
                h_t_kth_layer = self.__soft(matmul(h_t_1_last_layer, W_1) + matmul(self.input[t + 1], V), lambda1 / gama)
                h_t_all_layers = h_t_kth_layer

                for k in range(1, self.K):
                    # initialize V
                    V = one / gama * Ats[k]
                    # initialize W_1
                    W_1 = I - lambda2 / gama * AtAs[k]

                    W_k = lambda2 / gama * I
                    U_k = I - one / gama * AtAs[k]
                    h_t_kth_layer = self.__soft(tf.matmul(h_t_1_last_layer, W_k) * delta
                                         + matmul(h_t_kth_layer, U_k) \
                                         - matmul(h_t_kth_layer, W_k) * delta
                                         + matmul(self.input[t + 1], V), lambda1 / gama)
                    h_t_all_layers = tf.concat([h_t_all_layers, h_t_kth_layer], axis=1)
                h.append(h_t_all_layers)
                h_t_1_all_layers = h_t_all_layers
            h = tf.stack(h)

        return h, parameters

    def forward_coherence(self):
        '''
        :return: the hidden state with shape: [time_step, batch_size, K * n_hidden],
                 parameters: A dict with A, gama, lambda1, lambda2, h_0
        '''
        # define variables and constants in the sista_rnn model
        with tf.variable_scope('sista_rnn_trainable_variables'):
            A = self.__get_variable('A', None, self.A_initializer)
            gw = self.__get_variable('gw', None, self.gw_initializer)
            gama = self.__get_variable('gama', None, tf.constant(self.gama, tf.float32))
            lambda1 = self.__get_variable('lambda1', None, tf.constant(self.lambda1, tf.float32))
            lambda2 = self.__get_variable('lambda2', None, tf.constant(self.lambda2, tf.float32))
            h_0 = self.__get_variable('h_0', [self.batch_size, self.n_hidden], tf.zeros_initializer)
        parameters = OrderedDict([('A', A),
                                  ('gw', gw),
                                  ('gama', gama),
                                  ('lambda1', lambda1),
                                  ('lambda2', lambda2),
                                  ('h_0', h_0)])
        epsilon = tf.constant(100, dtype=tf.float32)
        At = tf.matrix_transpose(A)
        AtA = tf.matmul(At, A)
        I = tf.eye(self.n_hidden, self.n_hidden)
        one = tf.constant(1.0, dtype=tf.float32)

        # initialize V
        V = one / gama * At

        # initialize W_1
        W_1 = I - lambda2 / gama * AtA
        h_0_all_layers = tf.tile(h_0, [1, self.K])
        h_t_1_all_layers = h_0_all_layers

        # padding for the first time input
        if self.pre_input is None:
            self.input = tf.concat([tf.zeros([1, self.batch_size, self.n_input], tf.float32), self.now_input], axis=0)
        else:
            self.input = tf.concat([self.pre_input, self.now_input], axis=0)

        # sequential ISTA mapping to RNN
        with arg_scope([matmul], transpose_b=True):
            h = []

            gx1 = tf.nn.relu(dim_reduction(self.input[:-1], gw))
            gx2 = tf.nn.relu(dim_reduction(self.input[1:], gw))
            if len(gx1.get_shape().as_list()) != 3:
                gx1 = tf.expand_dims(gx1, 0)
                gx2 = tf.expand_dims(gx2, 0)
            gx1 = tf.nn.l2_normalize(gx1, dim=2)
            gx2 = tf.nn.l2_normalize(gx2, dim=2)
            # gx1 = self.input[:-1]
            # gx2 = self.input[1:]
            deltas = tf.tile(tf.reduce_sum(gx1 * gx2, axis=2, keep_dims=True), [1, 1, self.n_hidden])

            for t in range(self.time_steps):
                h_t_1_last_layer = h_t_1_all_layers[:, (self.K - 1) * self.n_hidden:]
                delta = deltas[t]
                h_t_kth_layer = self.__soft(matmul(h_t_1_last_layer, W_1) + matmul(self.input[t + 1], V), lambda1 / gama)
                h_t_all_layers = h_t_kth_layer

                for k in range(1, self.K):
                    W_k = lambda2 / gama * I
                    U_k = I - one / gama * AtA
                    h_t_kth_layer = self.__soft(tf.matmul(h_t_1_last_layer, W_k) * delta
                                         + matmul(h_t_kth_layer, U_k)
                                         - matmul(h_t_kth_layer, W_k) * delta
                                         + matmul(self.input[t + 1], V), lambda1 / gama)
                    h_t_all_layers = tf.concat([h_t_all_layers, h_t_kth_layer], axis=1)
                h.append(h_t_all_layers)
                h_t_1_all_layers = h_t_all_layers
            h = tf.stack(h)

        return h, parameters

    def forward_coherence_As(self):
        '''
        :return: the hidden state with shape: [time_step, batch_size, K * n_hidden],
                 parameters: A dict with A, gama, lambda1, lambda2, h_0
        '''
        # define variables and constants in the sista_rnn model
        As = []
        with tf.variable_scope('sista_rnn_trainable_variables'):
            for i in range(self.K):
                As.append(self.__get_variable('A_{}'.format(i + 1), None, self.A_initializer))
            gw = self.__get_variable('gw', None, self.gw_initializer)
            gama = self.__get_variable('gama', None, tf.constant(self.gama, tf.float32))
            lambda1 = self.__get_variable('lambda1', None, tf.constant(self.lambda1, tf.float32))
            lambda2 = self.__get_variable('lambda2', None, tf.constant(self.lambda2, tf.float32))
            h_0 = self.__get_variable('h_0', [self.batch_size, self.n_hidden], tf.zeros_initializer)
        parameters = OrderedDict([('A_{}'.format(i + 1), As[i]) for i in range(self.K)] + [
                                  ('gama', gama),
                                  ('lambda1', lambda1),
                                  ('lambda2', lambda2),
                                  ('h_0', h_0)])
        epsilon = tf.constant(100, dtype=tf.float32)
        Ats = [tf.matrix_transpose(As[i]) for i in range(self.K)]
        AtAs = [tf.matmul(Ats[i], As[i]) for i in range(self.K)]
        I = tf.eye(self.n_hidden, self.n_hidden)
        one = tf.constant(1.0, dtype=tf.float32)

        # initialize V
        V = one / gama * Ats[0]

        # initialize W_1
        W_1 = I - lambda2 / gama * AtAs[0]

        h_0_all_layers = tf.tile(h_0, [1, self.K])
        h_t_1_all_layers = h_0_all_layers

        # padding for the first time input
        if self.pre_input is None:
            self.input = tf.concat([tf.zeros([1, self.batch_size, self.n_input], tf.float32), self.now_input], axis=0)
        else:
            self.input = tf.concat([self.pre_input, self.now_input], axis=0)

        # sequential ISTA mapping to RNN
        with arg_scope([matmul], transpose_b=True):
            h = []

            gx1 = tf.nn.relu(dim_reduction(self.input[:-1], gw))
            gx2 = tf.nn.relu(dim_reduction(self.input[1:], gw))
            deltas = tf.tile(tf.reduce_sum(gx1 * gx2, axis=2, keep_dims=True), [1, 1, self.n_hidden])
            deltas = tf.nn.softmax(deltas, dim=0)

            for t in range(self.time_steps):
                h_t_1_last_layer = h_t_1_all_layers[:, (self.K - 1) * self.n_hidden:]
                delta = deltas[t]
                h_t_kth_layer = self.__soft(matmul(h_t_1_last_layer, W_1) + matmul(self.input[t + 1], V), lambda1 / gama)
                h_t_all_layers = h_t_kth_layer

                for k in range(1, self.K):
                    # initialize V
                    V = one / gama * Ats[k]
                    # initialize W_1
                    W_1 = I - lambda2 / gama * AtAs[k]

                    W_k = lambda2 / gama * I
                    U_k = I - one / gama * AtAs[k]
                    h_t_kth_layer = self.__soft(tf.matmul(h_t_1_last_layer, W_k) * delta
                                         + matmul(h_t_kth_layer, U_k) \
                                         - matmul(h_t_kth_layer, W_k) * delta
                                         + matmul(self.input[t + 1], V), lambda1 / gama)
                    h_t_all_layers = tf.concat([h_t_all_layers, h_t_kth_layer], axis=1)
                h.append(h_t_all_layers)
                h_t_1_all_layers = h_t_all_layers
            h = tf.stack(h)

        return h, parameters
