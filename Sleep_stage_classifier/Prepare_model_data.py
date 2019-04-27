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

import glob
import math
from sklearn.utils import class_weight
'''read data'''

shhs_base_dir = 'D:/Documents/GaTech/CSE 6250 Big data for Health/Term project/Ruby for download/shhs/polysomnography'
PATH_OUTPUT = "./output"
os.makedirs(PATH_OUTPUT, exist_ok=True)


def preparedata(sampling_rate,proportions = (0.5, 0.2, 0.3)):
    preprocessed_names = glob.glob(os.path.join(
    shhs_base_dir, 'edfs','shhs1', '*.edf'))
    nb_patients=math.ceil(len(preprocessed_names)*sampling_rate)
    r = np.arange(len(preprocessed_names))
    np.random.shuffle(r)
    preprocessed_names = [preprocessed_names[i] for i in r]
    preprocessed_names = preprocessed_names[:nb_patients]
    
    n_train = int(proportions[0]*len(preprocessed_names))
    print('n_train: ', n_train)
    n_valid = int(proportions[1]*len(preprocessed_names))
    print('n_valid: ', n_valid)
    names_train = preprocessed_names[0:n_train]
    names_valid = preprocessed_names[n_train:n_train+n_valid]
    names_test = preprocessed_names[n_train+n_valid:]

    for file in names_train:
        '''print('filename: ', os.path.basename(file))'''
        basename=os.path.basename(file).replace(".edf","")
        '''print('filename: ', basename)'''
        
        annotation=os.path.join(shhs_base_dir, 'annotations-events-nsrr','shhs1', basename+'-nsrr.xml')
        '''print('annotation: ', annotation)'''
        exists=os.path.isfile(annotation)
        if exists:
            raw = mne.io.read_raw_edf(file)
            annotations = pr.nsrr_sleep_stage_annotations(annotation)
            raw.set_annotations(annotations)
            events, _ = pr.sleep_stage_events(raw)
            event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3': 4,
                'Sleep stage R': 5}
            if len(set(events[:,2]))==5:
                epochs = pr.sleep_stage_epochs(raw=raw, events=events, event_id=event_id).load_data()
                x_train=eeg_power_band_shhs(epochs)
                x_train2=add_pre_post(x_train)
                y_train = (epochs.events[:, 2]-1)
                if x_train.shape[1]==12:                 
                    try:
                        _ = X_train.shape
                        X_train=np.concatenate((X_train, x_train2))
                        Y_train=np.concatenate((Y_train, y_train))
                    except NameError:
                        X_train=x_train2
                        Y_train=y_train
                
    for file in names_valid:
        '''print('filename: ', os.path.basename(file))'''
        basename=os.path.basename(file).replace(".edf","")
        '''print('filename: ', basename)'''
        
        annotation=os.path.join(shhs_base_dir, 'annotations-events-nsrr','shhs1', basename+'-nsrr.xml')
        '''print('annotation: ', annotation)'''
        exists=os.path.isfile(annotation)
        if exists:
            raw = mne.io.read_raw_edf(file)
            annotations = pr.nsrr_sleep_stage_annotations(annotation)
            raw.set_annotations(annotations)
            events, _ = pr.sleep_stage_events(raw)
            event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3': 4,
                'Sleep stage R': 5}
            if len(set(events[:,2]))==5:
                epochs = pr.sleep_stage_epochs(raw=raw, events=events, event_id=event_id).load_data()
                x_valid=eeg_power_band_shhs(epochs)
                x_valid2=add_pre_post(x_valid)
                y_valid = (epochs.events[:, 2]-1)
                if x_valid.shape[1]==12:                 
                    try:
                        _ = X_valid.shape
                        X_valid=np.concatenate((X_valid, x_valid2))
                        Y_valid=np.concatenate((Y_valid, y_valid))
                    except NameError:
                        X_valid=x_valid2
                        Y_valid=y_valid
                        
                
    for file in names_test:
        '''print('filename: ', os.path.basename(file))'''
        basename=os.path.basename(file).replace(".edf","")
        '''print('filename: ', basename)'''
        
        annotation=os.path.join(shhs_base_dir, 'annotations-events-nsrr','shhs1', basename+'-nsrr.xml')
        '''print('annotation: ', annotation)'''
        exists=os.path.isfile(annotation)
        if exists:
            raw = mne.io.read_raw_edf(file)
            annotations = pr.nsrr_sleep_stage_annotations(annotation)
            raw.set_annotations(annotations)
            events, _ = pr.sleep_stage_events(raw)
            event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3': 4,
                'Sleep stage R': 5}
            if len(set(events[:,2]))==5:
                epochs = pr.sleep_stage_epochs(raw=raw, events=events, event_id=event_id).load_data()
                x_test=eeg_power_band_shhs(epochs)
                x_test2=add_pre_post(x_test)
                y_test = (epochs.events[:, 2]-1)
                if x_test.shape[1]==12:                 
                    try:
                        _ = X_test.shape
                        X_test=np.concatenate((X_test, x_test2))
                        Y_test=np.concatenate((Y_test, y_test))
                    except NameError:
                        X_test=x_test2
                        Y_test=y_test
                
    return X_train,Y_train,X_valid,Y_valid,X_test,Y_test

X_train,Y_train,X_valid,Y_valid,X_test,Y_test=preparedata(0.02)

scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_valid=scaler.transform(X_valid)
X_test=scaler.transform(X_test)

np.save(os.path.join(PATH_TRAIN_FILE, 'X_train.npy'), X_train)
np.save(os.path.join(PATH_TRAIN_FILE, 'X_valid.npy'), X_valid)
np.save(os.path.join(PATH_TRAIN_FILE, 'X_test.npy'), X_test)
np.save(os.path.join(PATH_TRAIN_FILE, 'Y_train.npy'), Y_train)
np.save(os.path.join(PATH_TRAIN_FILE, 'Y_valid.npy'), Y_valid)
np.save(os.path.join(PATH_TRAIN_FILE, 'Y_test.npy'), Y_test)
