from __future__ import print_function
import keras
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, add
import tensorflow as tf
from keras.models import load_model
from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import os, glob, sys, threading
import scipy.io
from scipy import ndimage, misc
import numpy as np
import re
import math
import h5py
import imageio
import matplotlib.pyplot as plt


def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype)) #상용로그 정의
    return numerator / denominator


def load_h5(directory, filename, num):
    dirname=directory + filename
    with h5py.File(dirname, 'r') as hf:
        if filename[0] in ['r','g','b']:
            h5 = np.array(hf.get(filename[:-3]))
            h5 = np.transpose(h5, (2, 0, 1))
            h5 = np.reshape(h5,(h5.shape[0], h5.shape[1], h5.shape[2], 1))
        else:
            count = num
            s = 0
            while count>0:
                count = int(count/10)
                s = s+1
            h5 = np.array(hf.get(filename[s+1:-3]))
            h5 = np.transpose(h5)
            h5 = np.reshape(h5,(1, h5.shape[0], h5.shape[1],1))    
    return h5  


def PSNR(y_true, y_pred):
	max_pixel = 1.0
	return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) 


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def train_image_gen(patch, gt_patch, channel, BATCH_SIZE):
    patch = np.array(patch[channel])
    gt_patch = np.array(gt_patch[channel])
    while True:
        for step in range(len(patch)//BATCH_SIZE):
            offset = step*BATCH_SIZE
            batch_x = patch[offset:offset+BATCH_SIZE]/255
            batch_y = gt_patch[offset:offset+BATCH_SIZE]/255
            yield (batch_x, batch_y)
            
            
def valid_image_gen(img, gt_img, channel, num):
    offset = 100*channel
    while True:
        for i in range(num):
            batch_x = np.array(img[offset+i])/255
            batch_y = np.array(gt_img[offset+i])/255
            yield (batch_x, batch_y)
        

def step_decay(epoch):
    decay_rate = LEARNING_RATE/ (np.power(10,np.floor(epoch/20)))
    return decay_rate