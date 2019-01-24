# import pydevd
# pydevd.settrace('10.19.125.27', port=12345, stdoutToServer=True, stderrToServer=True)

import numpy as np
from concurrent import futures
from threading import Thread
import multiprocessing
import random
import time
import h5py
import sys
import os

pool_size = 4
scales = [1, 2, 4]
total_patches = 21
feature_size = 7.0

clip_length = 4
clip_feature = False


def readHDF5(video_idx, random_idx, key):
    with h5py.File(video_idx, 'r') as f:
        if clip_feature:
            #print(max(0, random_idx - clip_length // 2), min(f[key].shape[3], random_idx + clip_length / 2))
            #batch_features = np.concatenate((f[key][:, :, :, random_idx], np.mean(f[key][:, :, :, max(0, random_idx - clip_length // 2):min(f[key].shape[3], random_idx + clip_length // 2)], axis=3)), axis=0)
            batch_features = np.mean(f[key][:, :, :, max(0, random_idx - clip_length // 2):min(f[key].shape[3], random_idx + clip_length // 2)], axis=3)
            #batch_features = f[key][:, :, :, min(f[key].shape[3] - 1, random_idx + 1)] - f[key][:, :, :, random_idx]
        else:
            #batch_features = f[key][:, :, :, random_idx]
            batch_features = f[key][random_idx]
        #batch_features = np.transpose(batch_features, [2, 1, 0])
        multi_patch_features = np.zeros([total_patches, batch_features.shape[2]],dtype=np.float32)
        t = 0
        for s in scales:
            step = int(np.ceil(feature_size / s))
            for i in range(0, int(feature_size), step):
                for j in range(0, int(feature_size), step):
                    temp = batch_features[i:min(i+step,int(feature_size)), j:min(j+step,int(feature_size))]
                    if temp.ndim == 3:
                        multi_patch_features[t] = np.mean(np.mean(temp, axis=0), axis=0)
                    elif temp.ndim == 2:
                        multi_patch_features[t] = np.mean(temp, axis=0)
                    t = t + 1

        return multi_patch_features


class HDF5Reader(object):
    def __init__(self):
        pass

    def __call__(self, video_idx, random_idx, key):
        return readHDF5(video_idx, random_idx, key)

def advance_batch(result, sequence_generator, HDF5_reader, pool):
    random_idx, video_idx = sequence_generator()
    #debug
    # tmp1 = image_processor(images_path[0])
    # tmp2 = HDF5_reader(video_idx[0], random_idx[0])

    # result['rgb'] = HDF5_reader(video_idx[0], random_idx[0], 'rgb')
    # result['optical_flow'] = HDF5_reader(video_idx[0], random_idx[0], 'optical_flow')

    rgb_keys = ['rgb'] * len(video_idx)
    #optical_flow_keys = ['optical_flow'] * len(video_idx)
    rgb_feature = pool.map(HDF5_reader, video_idx, random_idx, rgb_keys)
    #optical_flow_feature = pool.map(HDF5_reader, video_idx, random_idx, optical_flow_keys)

    result['rgb'] = np.array(list(rgb_feature))
    #result['optical_flow'] = np.array(list(optical_flow_feature))

class BatchAdvancer():
    def __init__(self, result, sequence_generator, HDF5_reader, pool):
        self.sequence_generator = sequence_generator
        self.HDF5_reader = HDF5_reader
        self.pool = pool
        self.result = result

    def __call__(self):
        advance_batch(self.result, self.sequence_generator, self.HDF5_reader, self.pool)


class SequenceGeneratorVideo(object):
    def __init__(self, batch_size, clip_length, num_videos, video_dict, video_order):
        self.batch_size = batch_size
        self.clip_length = clip_length
        self.num_videos = num_videos
        self.N = self.clip_length * self.batch_size
        self.video_dict = video_dict
        self.video_order = video_order
        self.idx = 0

    def __call__(self):
        video_idx = [None] * self.batch_size * self.clip_length
        random_idx = [None] * self.batch_size * self.clip_length

        for i in range(0, self.batch_size):
            if self.idx >= self.num_videos:
                random.shuffle(self.video_order)
                self.idx = 0
            key = self.video_order[self.idx]
            random_start = int(random.random() * (self.video_dict[key]['num_frames'] - self.clip_length))
            k = 0
            for j in range(random_start, random_start + self.clip_length):
                video_idx[k * self.batch_size + i] = self.video_dict[key]['feature_hdf5_path']
                random_idx[k * self.batch_size + i] = j
                k += 1

            self.idx += 1

        return random_idx, video_idx


class FeatureLoader(object):
    def __init__(self, hdf5_path, videos_txt, batch_size, clip_length):
        self.hdf5_path = hdf5_path
        self.videos_txt = videos_txt
        self.batch_size = batch_size

        self.clip_length = clip_length
        self.setup()

    def setup(self):
        random.seed(2017)
        f = open(self.videos_txt, 'r')
        f_lines = f.readlines()
        f.close()

        # f_lines: dataset_train_01.h5
        #          dataset_train_02.h5
        #          xxx
        #          dataset_train_ab.h5

        # dataset_train_01.h5
        #          'video_length': length
        #          'rgb': []
        #          'optical_flow': []

        video_dict = {}
        self.video_order = []

        for idx, video in enumerate(f_lines):
            video = video.strip('\n')
            print(video)
            video_dict[video] = {}
            self.video_order.append(video)

            #dataset, video_name = video.split('.')[0].split('_train')
            feature_hdf5_path = os.path.join(self.hdf5_path, video)
            print(feature_hdf5_path)
            with h5py.File(feature_hdf5_path, 'r') as f:
                video_dict[video]['num_frames'] = f['rgb'].shape[0]
                video_dict[video]['feature_hdf5_path'] = feature_hdf5_path


        self.video_dict = video_dict
        self.num_videos = len(video_dict.keys())

        self.thread_result = {}
        self.thread = None

        self.sequence_generator = SequenceGeneratorVideo(self.batch_size, self.clip_length, self.num_videos,
                                                         self.video_dict, self.video_order)

        self.HDF5_reader = HDF5Reader()
        self.pool = futures.ProcessPoolExecutor(max_workers=pool_size)
        #self.pool = Pool(processes=pool_size)
        self.batch_advancer = BatchAdvancer(self.thread_result, self.sequence_generator, self.HDF5_reader, self.pool)
        #pre-load a batch
        self.dispatch_worker()
        self.join_worker()

    def load_batch(self):
        if self.thread is not None:
            self.join_worker()
        self.dispatch_worker()
        return self.thread_result


    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None

#demo
if __name__ == "__main__":

    # python libs/feature_loader.py 1
    # testing on simulated data
    if len(sys.argv) > 1:
        print('generating txt...')

        # dataset
        dataset = 'shanghaitech'
        # root path
        root = '/p300'
        # 152 or 50
        res_type = '152'

        for train_test in ['training', 'testing']:
            # feature path
            path = os.path.join(root, 'dataset/anomaly_detection', dataset, train_test, '224/features/twostream_res' + res_type + '_7x7')
            files = os.listdir(path)
            files.sort()
            name_videos = ''
            for file in files:
                name_videos += file + '\n'

            with open('../txt/' + dataset + '_feature_' + train_test + '.txt', 'w') as f:
                f.write(name_videos)

    # test demo
    else:
        feature_path = '/root/dataset/anomaly_detection/avenue/training/224/features/twostream_res152_7x7'
        video_txt = '../txt/avenue_feature_train.txt'

        feature_loader = FeatureLoader(feature_path, video_txt, batch_size=10, clip_length=4)

        for i in range(1000):
            batch = feature_loader.load_batch()

            rgb_feature = batch['rgb']

            print('rgb feature = {}'.format(rgb_feature.shape))
            #print('optical flow feature = {}'.format(optical_flow_feature.shape))
