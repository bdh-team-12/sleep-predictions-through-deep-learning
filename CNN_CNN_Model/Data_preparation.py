# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:01:45 2019

@author: CRNZ
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:42:48 2019

@author: CRNZ
"""

import numpy as np
import pandas as pd
import scipy.signal as ssignal
import os
import matplotlib.pyplot as plt 
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import psd_array_welch
from Utilities import eeg_power_band_shhs
import shhs.polysomnography.polysomnography_reader as pr
import glob
from glob import glob
import math
shhs_base_dir = 'D:/Documents/GaTech/CSE 6250 Big data for Health/Term project/Ruby for download/shhs/polysomnography'
output_dir="./data_npz"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
def preparedata(sampling_rate):
    preprocessed_names = glob(os.path.join(shhs_base_dir, 'edfs','shhs1', '*.edf'))
    nb_patients=math.ceil(len(preprocessed_names)*sampling_rate)
    r = np.arange(len(preprocessed_names))
    np.random.shuffle(r)
    preprocessed_names = [preprocessed_names[i] for i in r]
    preprocessed_names = preprocessed_names[:nb_patients]

    names_train = preprocessed_names

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
                x_train=eeg_power_band_shhs(epochs).transpose(0,2,1)
                y_train = (epochs.events[:, 2]-1)
        # Save
                filename = os.path.join(output_dir, basename+'.npz')
                save_dict = {
            "x": x_train, 
            "y": y_train, 
                }
                np.savez(filename, **save_dict)
    pass                    

preparedata(0.01)
