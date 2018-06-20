from __future__ import print_function
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, Merge, ZeroPadding2D, merge, add
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
            h5 = np.reshape(h5,(1, h5.shape[0], h5.shape[1],1))    
    return h5    

def PSNR(y_true, y_pred):
	max_pixel = 1.0
	return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) 

####################################################################################################################

def load_images(directory):
	images = []
	for root, dirnames, filenames in os.walk(directory):
	    for filename in filenames:
	        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
	            filepath = os.path.join(root, filename)
	            image = ndimage.imread(filepath, mode="L")
	            images.append(image)
	            
	images = np.array(images)
	array_shape = np.append(images.shape[0:3], 1)
	images = np.reshape(images, (array_shape))

	return images

def get_image_list(data_path):
    l = glob.glob(os.path.join(data_path,"*"))
    # print(len(l))
    l = [f for f in l if re.search("^\d+_\d+_gt.mat$", os.path.basename(f))]
    # print(len(l))
    r_train_list = []
    g_train_list = []
    b_train_list = []
    string = '.mat'
    count=0
    for f in l:        
        if os.path.exists(f):            
            if count%3 == 0:
                r_train_list.append([f, f[:-7]+string])
                count=count+1
            elif count%3 == 1:
                g_train_list.append([f, f[:-7]+string])
                count=count+1
            elif count%3 == 2:
                b_train_list.append([f, f[:-7]+string])
                count=count+1
    return r_train_list, g_train_list, b_train_list

#def get_valid_image

def get_image_batch(train_list, offset, BATCH_SIZE):
    target_list = train_list[offset:offset+BATCH_SIZE]
    input_list = []
    gt_list = []
    for pair in target_list:
        input_img = scipy.io.loadmat(pair[1])['patch']
        gt_img = scipy.io.loadmat(pair[0])['patch']        
        input_list.append(input_img)
        gt_list.append(gt_img)
    input_list = np.array(input_list)
    input_list.resize([BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1])
    gt_list = np.array(gt_list)
    gt_list.resize([BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1])
    return input_list, gt_list

def test_get_image_batch(train_list, offset, BATCH_SIZE):
	target_list = train_list[offset:offset+BATCH_SIZE]
	input_list = []
	gt_list = []
	for pair in target_list:
		input_img = scipy.io.loadmat(pair[1])['patch']
		gt_img = scipy.io.loadmat(pair[0])['patch']
		input_list.append(input_img)
		gt_list.append(gt_img)
	input_list = np.array(input_list)
	input_list.resize([BATCH_SIZE, input_img.shape[0], input_img.shape[1], 1])
	gt_list = np.array(gt_list)
	gt_list.resize([BATCH_SIZE, input_img.shape[0], input_img.shape[1], 1])
	return input_list, gt_list

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
def image_gen(target_list, BATCH_SIZE):
	while True:
		for step in range(len(target_list)//BATCH_SIZE):
			offset = step*BATCH_SIZE
			batch_x, batch_y = get_image_batch(target_list, offset, BATCH_SIZE)
			yield (batch_x, batch_y)
            
def test_image_gen(target_list, BATCH_SIZE):
	while True:
		for step in range(len(target_list)//BATCH_SIZE):
			offset = step*BATCH_SIZE
			batch_x, batch_y = test_get_image_batch(target_list, offset, BATCH_SIZE)
			yield (batch_x, batch_y)

def step_decay(epoch):
    decay_rate = LEARNING_RATE/ (np.power(10,np.floor(epoch/20)))
    return decay_rate

