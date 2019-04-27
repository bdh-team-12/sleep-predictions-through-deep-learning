# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 23:01:59 2019

@author: CRNZ
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:24:21 2019

@author: CRNZ
"""

import numpy as np
import pandas as pd
import scipy.signal as ssignal
import os
import matplotlib.pyplot as plt
"""this is from Blake's code"""     
import sys
sys.path.append("..")
import shhs.polysomnography.polysomnography_reader as pr
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import psd_array_welch
from utils import *
from sklearn.preprocessing import FunctionTransformer    
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pickle


def preparedata(filename):
    raw = mne.io.read_raw_edf(filename)
    a=int(len(raw)/3750)
    b=np.zeros(a)
    c=b+1
    d=np.linspace(0,a*3750,a,endpoint=False,dtype=int)
    e=np.vstack((d,b,c)).T.astype(int)
    event_id = {'Sleep stage W': 1}
    epochs = pr.sleep_stage_epochs(raw=raw, events=e, event_id=event_id).load_data()
    x_train=eeg_power_band_shhs(epochs)
    x_train2=add_pre_post(x_train)
                
    return x_train2


'''read data'''

filename = 'D:/Documents/GaTech/CSE 6250 Big data for Health/Term project/Ruby for download/shhs/polysomnography/edfs/shhs1/shhs1-200006.edf'
PATH_TRAIN_FILE = "./output"
PATH_MODEL = "./model/"
Score_output="./score_result"
os.makedirs(Score_output, exist_ok=True)

X_train=np.load(os.path.join(PATH_TRAIN_FILE, 'X_train.npy'))
X_score=preparedata(filename)
'''standardize based on X_train'''
scaler=StandardScaler()
scaler.fit(X_train)
X_score=scaler.transform(X_score)

np.save(os.path.join(PATH_TRAIN_FILE, 'X_score.npy'), X_score)
X_score=np.load(os.path.join(PATH_TRAIN_FILE, 'X_score.npy'))


save_file = 'RFC.sav'
RFC = pickle.load(open(os.path.join(PATH_MODEL, save_file), 'rb'))
YP_test=RFC.predict(X_score)
score_result = 'score.csv'
output_file = open(os.path.join(Score_output, score_result), 'w')
output_file.write("stages\n")
for y in YP_test:   
    output_file.write("{}\n".format(y))
output_file.close()