import os
import sys
import numpy as np
import yaml
import tensorflow as tf
from libs.sista_rnn_anomaly_detection import sista_rnn_anomaly_detection_TSC, sista_rnn_anomaly_detection_AE
from libs.common import checkdir
from libs import FLAGS
import multiprocessing

def main(config):
    if sys.argv[2] == '0':
        config['train_videos_txt'] = config['train_videos_txt'].format(config['dataset'])
        config['train_feature_path'] = config['train_feature_path'].format(config['dataset'], config['twostream_model'])
        config['prefix'] = config['prefix'].format(config['dataset'], config['twostream_model'], config['model_type'],
                                                   config['K'], config['n_hidden'], config['lambda1'], config['lambda2'], config['time_steps'])
        config['log_path'] = os.path.join(config['log_path'], config['prefix'])
        checkdir(config['ckpt_path'])
        checkdir(config['log_path'])
    elif sys.argv[2] == '1':
        config['time_steps'] = 1
        config['batch_size'] = 1
        config['test_videos_txt'] = config['test_videos_txt'].format(config['dataset'])
        config['test_feature_path'] = config['test_feature_path'].format(config['dataset'], config['twostream_model'])
        config['prefix'] = config['prefix'].format(config['dataset'], config['twostream_model'], config['model_type'],
                                                   config['K'], config['n_hidden'], config['lambda1'], config['lambda2'], 10)
        checkdir(config['save_results_path'])
        config['save_results_path'] = os.path.join(config['save_results_path'], config['prefix'])
    else:
        raise Exception('only support 0 for training or 1 for testing')
    rng = np.random.RandomState(config['seed'])

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    # ======================== SISTA-RNN ============================= #
    print('... building the sista-rnn networks')

    pre_input = tf.placeholder(tf.float32, [1, config['batch_size'] * 21, config['n_input']])
    now_input = tf.placeholder(tf.float32, [config['time_steps'], config['batch_size'] * 21, config['n_input']])
    A = np.asarray(
        rng.uniform(
            low=-np.sqrt(6.0 / (config['n_input'] + config['n_hidden'])),
            high=np.sqrt(6.0 / (config['n_input'] + config['n_hidden'])),
            size=(config['n_input'], config['n_hidden'])
        ) / 2.0,
        dtype=np.float32
    )

    if config['model_type'] == FLAGS.TSC:
        model = sista_rnn_anomaly_detection_TSC([pre_input, now_input], None, A, sess, config)
    elif config['model_type'] == FLAGS.AE:
        model = sista_rnn_anomaly_detection_AE([pre_input, now_input], None, A, sess, config)
    else:
        raise Exception('not support {}, only support TSC and AE model'.format(config['model_type']))

    if sys.argv[2] == '0':
        model.train()
    else:
        model.test()

if __name__ == '__main__':
    # ==========================load config================================ #
    if len(sys.argv) < 3:
        raise Exception('usage: python xxx.py config/xxx.yaml 0/1 (0 is training, 1 is testing)')
    with open(sys.argv[1], 'r') as stream:
        config = yaml.load(stream)


    class NoDaemonProcess(multiprocessing.Process):
        # make 'daemon' attribute always return False
        def _get_daemon(self):
            return False

        def _set_daemon(self, value):
            pass

        daemon = property(_get_daemon, _set_daemon)


    # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
    # because the latter is only a wrapper function, not a proper class.
    class NoDaemonProcessPool(multiprocessing.pool.Pool):
        Process = NoDaemonProcess

    pool = NoDaemonProcessPool(16)
    configs = []
    for index1, l1 in enumerate([0.3, 0.4, 0.5, 0.6]):
        for index2, l2 in enumerate([0.3, 0.4, 0.5, 0.6]):
            c = config.copy()
            c['lambda1'] = l1
            c['lambda2'] = l2
            c['gpu'] = str(index2%4)
            configs.append(c)
    #pool.map(main, configs)
    main(config)