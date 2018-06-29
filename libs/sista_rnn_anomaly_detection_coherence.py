from libs.base import base
from libs.sista_rnn import sista_rnn
from libs.feature_loader_multi_patch import FeatureLoader
import numpy as np
import tensorflow as tf
import os
import h5py
import pickle
from collections import OrderedDict
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.framework.python.ops.arg_scope import add_arg_scope
import time, datetime, csv

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

class sista_rnn_anomaly_detection(base):
    def __init__(self, input, label, A_initializer, gw_initializer, sess, config):
        self.A_initializer = A_initializer
        self.gw_initializer = gw_initializer
        self.sess = sess
        self.global_step = tf.Variable(0, trainable=False)
        super(sista_rnn_anomaly_detection, self).__init__(input, label, config)
        self.saver = tf.train.Saver(max_to_keep=None, var_list=self.optimized_parameters)

    def _head(self, input):
        return input, None

    def _body(self, input):
        model = sista_rnn(input[0], input[1], self.config['n_hidden'], self.config['K'], self.config['gama'], self.config['lambda1'], self.config['lambda2'], self.A_initializer, self.gw_initializer)
        h, parameters = model.forward_coherence()
        return h[..., -self.config['n_hidden']:], list(parameters.values())

    def _get_statistics(self):
        pred_reshape = tf.reshape(self.endpoints['body_output'], [-1, self.config['n_hidden']])
        loss2 = self.config['lambda1'] * tf.reduce_mean(tf.reduce_sum(tf.abs(pred_reshape), axis=1))

        gw = self.all_parameters['body_parameters'][1]
        x = self.endpoints['head_output'][1]
        gx1 = tf.nn.relu(dim_reduction(x[:-1], gw))
        gx2 = tf.nn.relu(dim_reduction(x[1:], gw))
        # gx1 = x[:-1]
        # gx2 = x[1:]
        if len(gx1.get_shape().as_list()) != 3:
            gx1 = tf.expand_dims(gx1, 0)
            gx2 = tf.expand_dims(gx2, 0)
        gx1 = tf.nn.l2_normalize(gx1, dim=2)
        gx2 = tf.nn.l2_normalize(gx2, dim=2)

        deltas = tf.reduce_sum(gx1 * gx2, axis=2)
        loss3 = self.config['lambda2'] * tf.reduce_mean(tf.reduce_sum(tf.square(self.endpoints['body_output'][:-1] - self.endpoints['body_output'][1:]), axis=2) * deltas)

        A_norm = tf.reduce_mean(tf.norm(self.all_parameters['body_parameters'][0], axis=0))
        h_nonzero_count = tf.reduce_mean(tf.count_nonzero(self.endpoints['body_output'], axis=2, dtype=tf.float32))

        self.gx = tf.nn.relu(dim_reduction(x, gw))

        return OrderedDict([('loss2', loss2),
                            ('loss3', loss3),
                            ('A_norm', A_norm),
                            ('h_nonzero_count', h_nonzero_count)])

    def train(self):
        # initialize a batch_loader
        featureLoader = FeatureLoader(self.config['train_feature_path'],
                                      self.config['train_videos_txt'],
                                      self.config['batch_size'],
                                      self.config['time_steps'])

        # define optimizer and gradients
        # optimizer = tf.train.MomentumOptimizer(config['learning_rate'], momentum=0.9)
        optimizer = tf.train.RMSPropOptimizer(self.config['learning_rate'])
        print(self.optimized_parameters)
        grad_var = optimizer.compute_gradients(self.total_loss, self.optimized_parameters)
        min_op = optimizer.apply_gradients(grad_var, self.global_step)
        self.sess.run(tf.global_variables_initializer())
        print('... training ...')

        iter = 0
        losses = []
        while iter < self.config['n_iter']:
            start = datetime.datetime.now()
            iter = iter + 1

            # batch.shape = NxD
            batch = featureLoader.load_batch()
            rgb = batch['rgb']
            rgb = np.reshape(rgb, [self.config['time_steps'], self.config['batch_size'] * 21, self.config['n_input']])
            pre_input = np.zeros([1, self.config['batch_size'] * 21, self.config['n_input']], dtype=np.float32)

            run_np = self.sess.run({'min_op': min_op,
                                    'global_step': self.global_step,
                                    'losses': self.losses,
                                    'summaries': self.summaries,
                                    'statistics': self.statistics},
                                   feed_dict={self.input[0]: pre_input,
                                              self.input[1]: rgb})

            if iter % self.config['display'] == 0:
                print(self.config['dataset'] + (' training iter = {} '
                                        + ''.join([name + ' = {} ' for name in self.losses.keys()])
                                        + ''.join([name + ' = {} ' for name in self.statistics.keys()])).
                      format(iter, *[value for value in run_np['losses'].values()], *[value for value in run_np['statistics'].values()]))

            if iter % self.config['snapshot'] == 0:
                self.saver.save(self.sess, os.path.join(self.config['ckpt_path'], self.config['prefix']), global_step=self.global_step)
                print('save model')

            if iter % self.config['summary'] == 0:
                self.summary_writer.add_summary(run_np['summaries'], run_np['global_step'])
                print('write summary')

            end = datetime.datetime.now()
            print((end - start).total_seconds())
            #print(list(run_np['losses'].values())[0])
            losses.append(list(run_np['losses'].values())[0])

        with open(str(self.config['K']) + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(losses)

    def test(self):
        self.sess.run(tf.global_variables_initializer())
        h_0 = self.all_parameters['body_parameters'][-1]
        reset_h_0_init_Tensor = tf.assign(h_0, tf.zeros([21, self.config['n_hidden']], dtype=tf.float32))
        update_h_0_Tensor = tf.assign(h_0, self.endpoints['body_output'][0])

        # load sista-rnn model parameters
        for test_loop in range(self.config['snapshot'], self.config['test_loop'] + self.config['snapshot'], self.config['snapshot']):
            ckpt_name = os.path.join(self.config['ckpt_path'], self.config['prefix'] + '-' + str(test_loop))
            while(not os.path.exists(ckpt_name+'.index')):
                print('I am sleeping', ckpt_name)
                time.sleep(100)
            self.saver.restore(self.sess, ckpt_name)

            print('... testing ...')

            with open(self.config['test_videos_txt'], 'r') as f:
                f_lines = f.readlines()

            total_loss = []

            for video in f_lines:
                video = video.strip('\n')
                f = h5py.File(os.path.join(self.config['test_feature_path'], video), 'r')
                rgb_data = np.transpose(f['rgb'], [3, 2, 1, 0])
                num_frames = rgb_data.shape[0]

                multi_patch_rgb_data = np.zeros([num_frames, 21, self.config['n_input']], dtype=np.float32)
                t = 0
                for s in self.config['scales']:
                    step = int(np.ceil(self.config['feature_size'] / s))
                    for i in range(0, int(self.config['feature_size']), step):
                        for j in range(0, int(self.config['feature_size']), step):
                            temp = rgb_data[:, i:min(i + step, int(self.config['feature_size'])),
                                   j:min(j + step, int(self.config['feature_size']))]
                            if temp.ndim == 4:
                                multi_patch_rgb_data[:, t] = np.mean(np.mean(temp, axis=1), axis=1)
                            elif temp.ndim == 3:
                                multi_patch_rgb_data[:, t] = np.mean(temp, axis=1)
                            t = t + 1

                sub_video_loss = np.zeros([num_frames, 21], dtype=np.float32)
                pre_input = np.zeros([1, self.config['batch_size'] * 21, self.config['n_input']], dtype=np.float32)
                h = self.endpoints['body_output'][0]
                for idx in range(num_frames):
                    rgb = np.expand_dims(multi_patch_rgb_data[idx], 0)
                    loss_np, _, h_np, gx_np = self.sess.run([self.testing_loss, update_h_0_Tensor, h, self.gx],
                                               feed_dict={self.input[0]: pre_input,
                                                          self.input[1]: rgb})
                    pre_input = rgb
                    sub_video_loss[idx] = loss_np

                    print('{} / {}, loss = {}'.format(video, idx, loss_np))

                # reset h0 for the next test video
                self.sess.run(reset_h_0_init_Tensor)

                # add to total loss
                if self.config['dataset'] == 'ped2':
                    temp = (sub_video_loss[:,0] + sub_video_loss[:,1:5].max(axis=1) + sub_video_loss[:,5:].max(axis=1)) / np.float32(3.0)
                    total_loss.append(temp)
                elif self.config['dataset'] == 'avenue' or self.config['dataset'] == 'shanghaitech':
                    total_loss.append(sub_video_loss[:, 0])

            results = {
                'dataset': self.config['dataset'],
                'mse': total_loss
            }

            with open(self.config['save_results_path'] + '_{}.bin'.format(test_loop), 'wb') as save_file:
                pickle.dump(results, save_file, 2)

            test_loop += self.config['snapshot']

class sista_rnn_anomaly_detection_AE(sista_rnn_anomaly_detection):
    def _tail(self, input):
        input_reshape = tf.reshape(input, [-1, self.config['n_hidden']])
        rng = np.random.RandomState(self.config['seed'])
        with tf.variable_scope('y'):
            Z = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6.0 / (self.config['n_input'] + self.config['n_hidden'])),
                    high=np.sqrt(6.0 / (self.config['n_input'] + self.config['n_hidden'])),
                    size=(self.config['n_output'], self.config['n_hidden'])
                ),
                dtype=np.float32
            )
            Z = tf.get_variable('Z', dtype=tf.float32, initializer=Z/2.0)
            y = tf.matmul(input_reshape, Z, transpose_b=True)

        return y, [Z]

    def _get_losses(self, pred, label):
        input_reshape = tf.reshape(self.endpoints['head_output'][1], [-1, self.config['n_input']])
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(input_reshape - pred), axis=1)) / 2.0

        return OrderedDict([('loss', loss)])

    def _get_optimized_parameters(self, all_parameters):
        return self.all_parameters['body_parameters'][:-2] + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'y')

    def _get_testing_loss(self, pred, label):
        input_reshape = tf.reshape(self.endpoints['head_output'][1], [-1, self.config['n_input']])
        loss = tf.reduce_sum(tf.square(input_reshape - pred), axis=1) / 2.0

        return loss

