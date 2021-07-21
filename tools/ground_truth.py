from __future__ import print_function
import scipy.io as scio
import numpy as np
import os
from libs import FLAGS


class GroundTruthLoader(object):
    AVENUE = FLAGS.AVENUE
    PED1 = FLAGS.PED1
    PED2 = FLAGS.PED2
    ENTRANCE = FLAGS.ENTRANCE
    EXIT = FLAGS.EXIT
    SHANGHAITECH = FLAGS.SHANGHAITECH
    MOVINGMNIST = FLAGS.MOVINGMNIST
    SHANGHAITEC_LABEL_PATH = './dataset/anomaly_detection/shanghaitech/testing/test_frame_mask/'

    NAME_MAT_MAPPING = {
        AVENUE: './dataset/anomaly_detection/gt_mat/avenue.mat',
        PED1: './dataset/anomaly_detection/gt_mat/ped1.mat',
        PED2: './dataset/anomaly_detection/gt_mat/ped2.mat',
        ENTRANCE: './dataset/anomaly_detection/gt_mat/enter_original.mat',
        EXIT: './dataset/anomaly_detection/gt_mat/exit.mat'
    }

    NAME_VIDEO_MAPPING = {
        AVENUE: './dataset/avenue/testing/videos',
        PED1: './dataset/ped1/testing/videos',
        PED2: './dataset/ped2/testing/videos',
        ENTRANCE: './dataset/enter/testing/videos',
        EXIT: './dataset/exit/testing/videos'
    }

    NAME_FRAMES_MAPPING = {
        AVENUE: './dataset/anomaly_detection/avenue/testing/frames',
        PED1: './dataset/anomaly_detection/ped1/testing/frames',
        PED2: './dataset/anomaly_detection/ped2/testing/frames',
        ENTRANCE: './dataset/anomaly_detection/enter/testing/frames',
        EXIT: './dataset/anomaly_detection/exit/testing/frames'
    }

    def __init__(self, mapping_json=None):
        if mapping_json is not None:
            import json
            with open(mapping_json, 'r') as f:
                self.mapping = json.load(f)

        else:
            self.mapping = GroundTruthLoader.NAME_MAT_MAPPING

    def __call__(self, dataset=AVENUE):
        """ get the ground truth by provided the name of dataset.

        :type dataset: str
        :param dataset: the name of dataset.
        :return: np.ndarray, shape(#video)
                 np.array[0] contains all the start frame and end frame of abnormal events of video 0,
                 and its shape is (#framse, )
        """

        if dataset == GroundTruthLoader.SHANGHAITECH:
            gt = self.__load_shanghaitech_gt()
        elif dataset == GroundTruthLoader.MOVINGMNIST:
            gt = self.__load_moving_mnist_gt()
        else:
            gt = self.__load_ucsd_avenue_subway_gt(dataset)
        return gt

    def __load_ucsd_avenue_subway_gt(self, dataset):
        assert dataset in self.mapping, 'there is no dataset named {} \n Please check {}' \
            .format(dataset, GroundTruthLoader.NAME_MAT_MAPPING.keys())

        mat_file = self.mapping[dataset]
        abnormal_events = scio.loadmat(mat_file, squeeze_me=True)['gt']

        if abnormal_events.ndim == 2:
            abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0], abnormal_events.shape[1])

        num_video = abnormal_events.shape[0]
        dataset_video_folder = GroundTruthLoader.NAME_FRAMES_MAPPING[dataset]
        video_list = os.listdir(dataset_video_folder)

        assert num_video == len(video_list), 'ground true does not match the number of testing videos. {} != {}' \
            .format(num_video, len(video_list))

        # video name template
        # /path/datasets/0...xx/
        video_name_template = os.path.join(dataset_video_folder, '{:0>%d}' % (len(str(num_video))))

        # video_name_template = os.path.join(dataset_video_folder, dataset + '_test_{0:02d}'.format(len(str(num_video))))

        # get the total frames of sub video
        def get_video_length(sub_video_number):
            video_name = video_name_template.format(sub_video_number)
            assert os.path.isdir(video_name), '{} is not directory!'.format(video_name)

            length = len(os.listdir(video_name))

            return length

        # need to test [].append, or np.array().append(), which one is faster
        gt = []
        for i in range(num_video):
            length = get_video_length(i + 1)

            sub_video_gt = np.zeros((length), dtype=np.int8)
            sub_abnormal_events = abnormal_events[i]
            if sub_abnormal_events.ndim == 1:
                sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))

            _, num_abnormal = sub_abnormal_events.shape

            for j in range(num_abnormal):
                # (start - 1, end - 1)
                start = sub_abnormal_events[0, j] - 1
                end = sub_abnormal_events[1, j]

                sub_video_gt[start: end] = 1

            gt.append(sub_video_gt)

        return gt

    def __load_shanghaitech_gt(self):
        video_path_list = os.listdir(GroundTruthLoader.SHANGHAITEC_LABEL_PATH)
        video_path_list.sort()

        gt = []
        for video in video_path_list:
            # print(os.path.join(GroundTruthLoader.SHANGHAITEC_LABEL_PATH, video))
            gt.append(np.load(os.path.join(GroundTruthLoader.SHANGHAITEC_LABEL_PATH, video)))

        return gt

    def __load_moving_mnist_gt(self):
        # label = np.load('/home/luowx/datasets/anomaly_detection/gt_mat/mnist_occlusion_test_label.npy')
        label = np.load('/home/luowx/datasets/anomaly_detection/gt_mat/mnist_motion_test_label.npy')
        label = np.squeeze(label)
        gt = []
        for i in range(100):
            gt.append(np.array(label[i * 20:(i + 1) * 20], dtype=np.int))
        return gt
