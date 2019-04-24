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
import shhs.polysomnography.polysomnography_reader as ps
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import psd_array_welch

from sklearn.preprocessing import FunctionTransformer    
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pickle
def eeg_power_band_shhs(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    '''0.75-4.50 Hz (Delta), 4.75-7.75 (Theta), 8.00-12.25 (Alpha), 12.50-15.00 (Sigma), 15.25-24.75 (Beta), 25.00-34.75 (Gamma 1), and 35.00-44.75 (Gamma 2) The '''
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30],
                  "Gamma":[30,50]}
    '''FREQ_BANDS = {"delta1": [0.5, 2.5],
                  "delta2": [2.5, 4.5],
                  "theta1": [4.5, 6.5],
                  "theta2": [6.5, 8.5],                  
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30],
                  "Gamma":[30,50]}'''
    EEG_CHANNELS = ["EEG(sec)", "EEG"]

    sfreq = epochs.info['sfreq']
    data = epochs.load_data().pick_channels(EEG_CHANNELS).get_data()
    psds, freqs = psd_array_welch(data, sfreq, fmin=0.5, fmax=50.,
                                  n_fft=512, n_overlap=256)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for _, (fmin, fmax) in FREQ_BANDS.items():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)    

def add_pre_post(data):
    '''EEG relative power band feature extraction'''
    df=pd.DataFrame(data)
    df2=df.shift(1).fillna(0)
    df3=df.shift(-1).fillna(0)
    df4=df.shift(2).fillna(0)
    df5=df.shift(-2).fillna(0)
    df_diff1=df.subtract(df2, fill_value=0)
    df_diff2=df.subtract(df3, fill_value=0)   
    dffinal=pd.concat([df,df2,df3], axis=1)
    return dffinal.values


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
PATH_TRAIN_FILE = "../output"
MODEL_TYPE = 'RFC'
PATH_OUTPUT = "../output/model/"
Score_output="../output"


X_train=np.load(os.path.join(PATH_TRAIN_FILE, 'X_train.npy'))
X_score=preparedata(filename)
'''standardize based on X_train'''
scaler=StandardScaler()
scaler.fit(X_train)
X_score=scaler.transform(X_score)

np.save(os.path.join(PATH_TRAIN_FILE, 'X_score.npy'), X_score)
X_score=np.load(os.path.join(PATH_TRAIN_FILE, 'X_score.npy'))


if MODEL_TYPE == 'RFC':
    save_file = 'RFC.sav'
    RFC = pickle.load(open(os.path.join(PATH_OUTPUT, save_file), 'rb'))
    YP_test=RFC.predict(X_score)
    score_result = 'score.csv'
    output_file = open(os.path.join(Score_output, score_result), 'w')
    output_file.write("stages\n")
    for y in YP_test:
        output_file.write("{}\n".format(y))
    output_file.close()