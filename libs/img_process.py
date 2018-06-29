import numpy as np
from scipy.ndimage import imread
from skimage.transform import resize
import glob
import os
import cv2
import threading

np.random.RandomState(2017)


##
# Data
##
def normalize_frames(frames):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].

    @param frames: A numpy array. The frames to be converted.

    @return: The normalized frames.
    """
    new_frames = frames.astype(np.float32)
    new_frames /= 255.
    # new_frames -= 1
    # new_frames /= 255   # for flownet
    return new_frames


def denormalize_frames(frames):
    """
    Performs the inverse operation of normalize_frames.

    @param frames: A numpy array. The frames to be converted.

    @return: The denormalized frames.
    """
    new_frames = frames + 1
    new_frames *= (255 / 2)
    # noinspection PyUnresolvedReferences
    # new_frames = frames * 255   # for flownet
    new_frames = new_frames.astype(np.uint8)

    return new_frames


def process_frame(image_name, height, width, norm):
    image = cv2.imread(image_name)
    image = cv2.resize(image, (height, width))
    if norm:
        image = normalize_frames(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class DataLoader(object):
    def __init__(self, directory, is_norm, height=224, width=224, num_classes=None, train_test_list_path=None):
        self.dir = directory
        self.videos = {}
        self.videos_names = None
        self.num_videos = 0
        self.height = height
        self.width = width
        self.is_norm = is_norm
        self.num_classes = num_classes

        # if class_id_path:
        #     self.class_name2id, self.class_id2name = self._load_class_id(class_id_path)
        #     self.num_classes = len(self.class_name2id.keys())
        # else:
        #     self.class_name2id, self.class_id2name = dict(), dict()
        #     self.num_classes = -1

        self.videos_labels = self._load_train_test_list_path(directory, train_test_list_path)

        self.setup()

    def __getitem__(self, video_name):
        assert video_name in self.videos.keys(), 'video = {} is not in {}!'.format(video_name, self.videos.keys())
        return self.videos[video_name]

    def __str__(self):
        _str = ('Video Directory = {}\n' +
                '   total videos = {}\n' +
                '   normalization = {}\n' +
                '   height x width = {} x {}')\
            .format(self.dir, self.num_videos, self.is_norm, self.height, self.width)
        return _str

    # @staticmethod
    # def _load_class_id(class_id_path):
    #     assert os.path.exists(class_id_path), 'class_id path {} is not exist!'.format(class_id_path)
    #     class_name2id = dict()
    #     class_id2name = dict()
    #     with open(class_id_path, 'r') as file:
    #         for row in file:
    #             row = row.strip()
    #             row = row.rstrip()
    #             splits = row.split(' ')
    #             video, label = splits[1], int(splits[0]) - 1
    #             class_name2id[video] = label
    #             class_id2name[label] = video
    #     return class_name2id, class_id2name

    @staticmethod
    def _load_train_test_list_path(video_dir, file_path):
        assert os.path.isdir(video_dir), 'video_dir {} is not a directory!'.format(video_dir)
        videos = os.listdir(os.path.join(video_dir))
        videos_labels = []
        if file_path:
            with open(file_path, 'r') as file:
                for row in file:
                    row = row.strip()
                    row = row.rstrip()
                    splits = row.split(' ')
                    if len(splits) == 2:
                        video, video_label = splits[0], int(splits[1]) - 1
                    else:
                        video, video_label = splits[0], -1

                    # ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi
                    video = video.split('/')[-1].split('.')[0]
                    assert videos.count(video) == 1, 'video {} != 1'.format(video)
                    videos_labels.append((video, video_label))
        else:
            for video in videos:
                videos_labels.append((video, -1))
        return videos_labels

    # def get_video_label(self, video_name):
    #     class_id = -1
    #     for name, c_id in self.class_name2id.items():
    #         if name.lower() in video_name.lower():
    #             class_id = c_id
    #             break
    #     # assert class_id != -1, '{} do not have class id {}!'.format(video_name, class_id)
    #     return class_id

    def setup(self):
        self.num_videos = len(self.videos_labels)
        for video, label in self.videos_labels:
            video_path = os.path.join(self.dir, video)
            self.videos[video] = {}
            self.videos[video]['path'] = video_path
            self.videos[video]['frame'] = glob.glob(os.path.join(video_path, '*.jpg'))
            self.videos[video]['frame'].sort()
            self.videos[video]['length'] = len(self.videos[video]['frame'])
            # if self.videos[video]['length'] == 0:
            #     with open('empty_list.txt', 'w') as file:
            #         shutil.rmtree(video_path)
            #         print(video_path)

            self.videos[video]['index'] = -1
            self.videos[video]['label'] = label

        self.videos_names = list(self.videos.keys())

    def get_video_info(self, video):
        return self.videos[video]

    def load_batch_start_step(self, video, length, step):
        # if read to the end of a video, then return None
        if self.videos[video]['index'] + length >= self.videos[video]['length']:
            return None
        images = []
        for i in range(length):
            index = self.videos[video]['index'] + i + 1
            image = process_frame(self.videos[video]['frame'][index], self.height, self.width, self.is_norm)
            images.append(image)

        self.videos[video]['index'] += step
        return np.expand_dims(np.concatenate(images, axis=2), axis=0)

    def load_batch_start_end(self, video, start, end):
        assert video in self.videos, 'video = {} must in {}!'.format(video, self.videos.keys())
        assert start >= 0, 'start = {} must >=0!'.format(start)
        assert end <= self.videos[video]['length'], 'end = {} must <= {}'.format(video, self.videos[video]['length'])

        batch = []
        for i in range(start, end):
            image = process_frame(self.videos[video]['frame'][i], self.height, self.width, self.is_norm)
            batch.append(image)

        return np.expand_dims(np.concatenate(batch, axis=2), axis=0)

    def load_frame(self, video):
        if self.videos[video]['index'] >= self.videos[video]['length'] - 1:
            return None
        index = self.videos[video]['index'] + 1
        self.videos[video]['index'] += 1
        image = process_frame(self.videos[video]['frame'][index], self.height, self.width, self.is_norm)
        return image

    def load_random_batch(self, batch_size, length):
        batch_images = []
        for video_name in np.random.choice(self.videos_names, batch_size, replace=False):
            video = self.get_video_info(video_name)
            start = np.random.choice(video['length'] - length + 1)
            images = []
            for frame_id in range(start, start+length):
                image = process_frame(video['frame'][frame_id], self.height, self.width, self.is_norm)
                images.append(image)
            images = np.concatenate(images, axis=2)
            batch_images.append(images)
        batch_images = np.stack(batch_images)
        return batch_images

    def load_random_batch_with_label(self, batch_size, length):
        batch_images = []
        batch_labels = np.zeros((length * batch_size, self.num_classes), dtype=np.uint8)
        for b_id, video_name in enumerate(np.random.choice(self.videos_names, batch_size)):
            video = self.get_video_info(video_name)
            class_id = video['label']
            batch_labels[b_id:-1:batch_size, class_id] = 1

            start = np.random.choice(video['length'] - length + 1)
            images = []
            for frame_id in range(start, start+length):
                image = process_frame(video['frame'][frame_id], self.height, self.width, self.is_norm)
                images.append(image)
            images = np.stack(images)
            batch_images.append(images)

        batch_images = np.stack(batch_images, 1)
        return batch_images, batch_labels

    def load_sequence_batch(self, batch_size, length):
        batch_images = []
        for video_name in np.random.choice(self.num_videos, batch_size):
            video = self.get_video_info(video_name)
            start = video['index'] + 1
            images = []

            for frame_id in range(start, start + length):
                frame_id = frame_id % video['length']
                image = process_frame(video['frame'][frame_id], self.height, self.width, self.is_norm)
                images.append(image)

            video['index'] = (start + length) % video['length']

            images = np.concatenate(images, axis=2)
            batch_images.append(images)
        batch_images = np.stack(batch_images)
        return batch_images

    def load_random_batch_sequence(self, batch_size, seq_len, interval):
        itv = np.random.randint(1, interval + 1)
        batch_sequence = np.zeros(shape=[batch_size, seq_len, self.height, self.width, 3], dtype=np.float32)
        for b_idx, video_name in enumerate(np.random.choice(self.videos_names, batch_size)):
            video = self.get_video_info(video_name)

            t = itv if seq_len * itv <= video['length'] else 1
            length = seq_len * t
            start = np.random.randint(video['length'] - length + 1)

            for t_idx, frame_id in enumerate(range(start, start+length, t)):
                image = process_frame(video['frame'][frame_id], self.height, self.width, self.is_norm)
                batch_sequence[b_idx, t_idx, ...] = image

        return batch_sequence


class TFLoaderThread(threading.Thread):

    def __init__(self, thread_name, sess, enqueue_op, coord, data_input, data_loader, batch_size, length=1):
        self.sess = sess
        self.enqueue_op = enqueue_op
        self.coord = coord
        self.data_input = data_input
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.length = length
        self.is_run = True
        threading.Thread.__init__(self, name=thread_name)

    def run(self):
        with self.coord.stop_on_exception():
            while not self.coord.should_stop():
                # batch_data = self.data_loader.load_random_batch(self.batch_size, self.length)
                batch_data = np.zeros((self.batch_size, 224, 224, 3))
                self.sess.run(self.enqueue_op, feed_dict={
                    self.data_input: batch_data
                })

                if not self.is_run:
                    self.coord.request_stop()


def singleton(cls, *args, **kwargs):
    instances = {}

    def _singleton():
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton


if __name__ == '__main__':
    data_loader = DataLoader('/home/luowx/datasets/UCF101_frames_fps10', height=224, width=224,
                             is_norm=True,
                             num_classes=101,
                             train_test_list_path='/home/luowx/datasets/ucfTrainTestlist/trainlist01.txt')
    data, label = data_loader.load_random_batch_with_label(4, 10)

