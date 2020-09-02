from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
# from fr_utils import *
# from inception_blocks_v2 import *
# import imutils
# from FaceDetector import *
from keras.models import model_from_json
import keras
# from generator_utils import *
from utility import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import time
# from parameters import *
ALPHA = 0.5
THRESHOLD = 0.5
IMAGE_SIZE= 96
LAYERS_TO_FREEZE= 60
NUM_EPOCHS= 5
STEPS_PER_EPOCH= 3
BATCH_SIZE= 64

IMAGE_SIZE=160
best_model_path="models/facenet_keras.h5"
def triplet_loss(y_true, y_pred, alpha = ALPHA):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    print("here")
    return loss



FRmodel = keras.models.load_model(best_model_path, custom_objects={'triplet_loss': triplet_loss})

for layers in FRmodel.layers:
    layers.trainable= False
# for layer in FRmodel.layers[0: LAYERS_TO_FREEZE]:
#     layer.trainable  =  False
    
    
# input_shape=(3, IMAGE_SIZE, IMAGE_SIZE)
input_shape=(IMAGE_SIZE, IMAGE_SIZE,3)
A = Input(shape=input_shape, name = 'anchor')
P = Input(shape=input_shape, name = 'anchorPositive')
N = Input(shape=input_shape, name = 'anchorNegative')

enc_A = FRmodel(A)
enc_P = FRmodel(P)
enc_N = FRmodel(N)

#
# # Callbacks
# early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0.00005)
# STAMP = 'facenet_%d'%(len(paths))
# checkpoint_dir = './' + 'checkpoints/' + str(int(time.time())) + '/'
#
# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir)
#
# bst_model_path = checkpoint_dir + STAMP + '.h5'
# tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

# Model
tripletModel = Model([A, P, N],[enc_A, enc_P, enc_N]) # input, output
tripletModel.compile(optimizer = 'adam', loss = triplet_loss)

# gen = batch_generator(BATCH_SIZE)
img1 = cv2.imread("images/chris/Positive_Chris.jpg", 1)
print("Positive:", img1.shape)
img2 = cv2.imread("images/chris/Anchor-Chris.jpg", 1)
print("Anchor:", img1.shape)
img3 = cv2.imread("images/chris/Negative-chris.jpg", 1)
print("Negative:", img1.shape)
# resize the image to 96 x 96
# img1 = cv2.resize(img1, (96, 96))

img1 = cv2.resize(img1, (160, 160))
img2 = cv2.resize(img2, (160, 160))
img3 = cv2.resize(img3, (160, 160))
# img = img1[...,::-1] # skipping for img1 = cv2.resize(img1, (96, 96))
# img=img1
print("this:", img1.shape)
# img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12) #skipping for facenet
# print("After transpose:",img.shape)
x_train = [np.array([img1]), np.array([img2]), np.array([img3])]
gen=x_train
# tripletModel.fit_generator(gen, epochs=NUM_EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,callbacks=[early_stopping, tensorboard])
tripletModel.fit_generator(gen, epochs=NUM_EPOCHS,steps_per_epoch=STEPS_PER_EPOCH)
bst_model_path="models//New//"
tripletModel.save(bst_model_path)
with open('bestmodel.txt','w') as file:
    file.write(bst_model_path)