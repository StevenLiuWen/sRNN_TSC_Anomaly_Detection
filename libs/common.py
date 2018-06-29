import os
import tensorflow as tf

def checkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def checkrank(tensor, ranks):
    for rank in ranks:
        if tf.rank(tensor) != rank:
            raise Exception('the rank of tensor {} is not equal to anyone of {}'.format(tensor, rank))