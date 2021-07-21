from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from sklearn import metrics
from six.moves import cPickle
from tools.ground_truth import GroundTruthLoader
import glob


class RecordResult(object):
    def __init__(self, fpr=None, tpr=None, auc=-np.inf, dataset=None, loss_file=None):
        self.fpr = fpr
        self.tpr = tpr
        self.auc = auc
        self.dataset = dataset
        self.loss_file = loss_file

    def __lt__(self, other):
        return self.auc < other.auc

    def __str__(self):
        return 'dataset = {}, loss file = {}, auc = {}'.format(self.dataset, self.loss_file, self.auc)


def parser_args():
    parser = argparse.ArgumentParser(description='evaluating the model, computing the roc/auc.')

    parser.add_argument('--file', type=str, help='the path of loss file.')
    parser.add_argument('--type', type=str, default='compute_auc', help='the type of evaluation, '
                                                                        'choosing type is: plot_roc, compute_auc, test_func\n, the default type is compute_auc')
    return parser.parse_args()


def load_loss_gt(loss_file):
    with open(loss_file, 'rb') as f:
        # results {
        #   'dataset': the name of dataset
        #   'mse': the mse of each testing videos,
        # }

        # mse_records['mse'] is np.array, shape(#videos)
        # mse_records[0] is np.array   ------>     01.avi
        # mse_records[1] is np.array   ------>     02.avi
        #               ......
        # mse_records[n] is np.array   ------>     xx.avi

        results = cPickle.load(f)

    dataset = results['dataset']
    mse_records = results['mse']

    num_videos = len(mse_records)

    # load ground truth
    gt_loader = GroundTruthLoader()
    gt = gt_loader(dataset=dataset)

    assert num_videos == len(gt), 'the number of saved videos does not match the ground truth, {} != {}' \
        .format(num_videos, len(gt))

    return dataset, mse_records, gt


def plot_roc(loss_file):
    optimal_results = compute_auc(loss_file)

    # the name of dataset, loss, and ground truth
    dataset, mse_records, gt = load_loss_gt(loss_file=optimal_results.loss_file)

    # the number of videos
    num_videos = len(mse_records)

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    sub_video_length_list = []

    # video normalization
    for i in range(num_videos):
        distance = mse_records[i]

        # distance -= distance.min()    # distances = (distance - min) / (max - min)
        # distance /= distance.max()
        # distance = 1 - distance

        distance = (distance - distance.min()) / (distance.max() - distance.min())
        distance = 1 - distance

        # print(distance.max())
        # print(distance.min())

        scores = np.concatenate((scores, distance), axis=0)
        labels = np.concatenate((labels, gt[i]), axis=0)

        sub_video_length_list.append(gt[i].shape[0])

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)

    # np.savetxt('ped2_scores.txt', scores)
    # np.savetxt('ped2_labels.txt', labels)

    # plot the scores
    total = scores.shape[0]
    index = range(0, total, 100)
    # plt.plot(index, scores[index], color='blue')
    plt.plot(scores, color='blue')

    # plot the ground truth
    i = 0
    while i < total:
        if labels[i] == 1:
            start = i
            end = i
            while end < total and labels[end] == 1:
                end += 1
            currentAxis = plt.gca()
            currentAxis.add_patch(Rectangle((start, 0.0), end - start, 1.0, color='red', alpha=0.3))

            i = end
        else:
            i += 1

    # plot line of different videos
    cur_len = 0
    for length in sub_video_length_list:
        cur_len += length
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((cur_len, 0.0), 1, 1.0, color='green', alpha=1))

    plt.annotate('AUC = ' + str(auc),
                 xy=(total / 2, 0.5), xycoords='data',
                 xytext=(-90, -50), textcoords='offset points', fontsize=15,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.title(dataset)
    plt.ylim(0)
    plt.xlabel('#Frames')
    plt.ylabel('Scores')
    plt.show()

    return optimal_results


def compute_auc(loss_file):
    # if not os.path.isdir(loss_file):
    #    loss_file_list = [loss_file]
    # else:
    #    loss_file_list = os.listdir(loss_file)
    loss_file_list = glob.glob(loss_file)
    # loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]
    print(loss_file_list)
    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        # the name of dataset, loss, and ground truth
        dataset, mse_records, gt = load_loss_gt(loss_file=sub_loss_file)

        # the number of videos
        num_videos = len(mse_records)

        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        # video normalization
        for i in range(num_videos):
            distance = mse_records[i]

            distance -= distance.min()  # distances = (distance - min) / (max - min)
            distance /= distance.max()
            distance = 1 - distance
            # distance = distance[:, 0]

            print(scores.shape, distance.shape)
            scores = np.concatenate((scores, distance), axis=0)
            labels = np.concatenate((labels, gt[i]), axis=0)

        if np.isnan(scores).any():
            continue
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        auc = metrics.auc(fpr, tpr)
        # eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        # thresh = interp1d(fpr, thresholds)(eer)
        results = RecordResult(fpr, tpr, auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        print(results)
        # with open(sub_loss_file+'.csv', 'w') as f:
        #    writer = csv.writer(f)
        #    writer.writerow(fpr)
        #    writer.writerow(tpr)

    return optimal_results


def compute_eer(loss_file):
    pass


def test_func(*args):
    # simulate testing on CUHK AVENUE dataset
    dataset = GroundTruthLoader.AVENUE

    # load the ground truth
    gt_loader = GroundTruthLoader()
    gt = gt_loader(dataset=dataset)

    num_videos = len(gt)

    simulated_results = {
        'dataset': dataset,
        'mse': []
    }

    simulated_mse = []
    for i in range(num_videos):
        sub_video_length = gt[i].shape[0]
        simulated_mse.append(np.random.random(size=sub_video_length))

    simulated_results['mse'] = simulated_mse

    # writing to file, 'generated_loss.bin'
    with open('generated_loss.bin', 'wb') as save_file:
        cPickle.dump(simulated_results, save_file, cPickle.HIGHEST_PROTOCOL)

    print(save_file.name)
    auc, dataset = plot_roc(save_file.name)

    print('optimal! dataset = {}, auc = {}'.format(dataset, auc))


eval_type_function = {
    'compute_auc': compute_auc,
    'compute_eer': compute_eer,
    'plot_auc': plot_roc
}


def evaluate(eval_type, save_file):
    assert eval_type in eval_type_function, 'there is no type of evaluation {}, please check {}' \
        .format(eval_type, eval_type_function.keys())

    eval_func = eval_type_function[eval_type]

    optimal_results = eval_func(save_file)

    print('dataset = {}, auc = {}'.format(optimal_results.dataset, optimal_results.auc))


if __name__ == '__main__':
    args = parser_args()

    eval_type = args.type
    save_file = args.file
    print(save_file)
    if eval_type == 'test_func':
        test_func()
    else:
        evaluate(eval_type, save_file)
