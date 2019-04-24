# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:11:51 2019

@author: CRNZ
"""
import numpy as np
import pandas as pd
import glob
from glob import glob
from Models import get_base_model,get_model_cnn
from keras import optimizers, losses, activations, models
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate, SpatialDropout1D, TimeDistributed, Bidirectional, LSTM
from keras_contrib.layers import CRF
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from Utilities import *

data_path = "./data_npz"
files = sorted(glob(os.path.join(data_path, "*.npz")))
file_path = "cnn_model.h5"
epochs=30

ids = sorted(list(set([x.split("\\")[-1][:12] for x in files])))
#split by test subject
train_ids, test_ids = train_test_split(ids, test_size=0.15, random_state=1338)

train_val, test = [x for x in files if x.split("\\")[-1][:12] in train_ids],\
                  [x for x in files if x.split("\\")[-1][:12] in test_ids]

train, val = train_test_split(train_val, test_size=0.1, random_state=1337)

train_dict = {k: np.load(k) for k in train}
test_dict = {k: np.load(k) for k in test}
val_dict = {k: np.load(k) for k in val}

model = get_model_cnn()


# model.load_weights(file_path)

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=20, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=5, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

model.fit_generator(gen(train_dict, aug=False), validation_data=gen(val_dict), epochs=epochs, verbose=2,
                    steps_per_epoch=1000, validation_steps=300, callbacks=callbacks_list)
model.load_weights(file_path)
