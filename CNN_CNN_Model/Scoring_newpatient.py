# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:52:09 2019

@author: CRNZ
"""

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
from Models import get_base_model,get_model_cnn
import shhs.polysomnography.polysomnography_reader as pr
import glob
from glob import glob
import math
from tqdm import tqdm
from Utilities import *
from pathlib import Path
filename = 'D:/Documents/GaTech/CSE 6250 Big data for Health/Term project/Ruby for download/shhs/polysomnography/edfs/shhs1/shhs1-200006.edf'    

output_dir=Path('./score_result')
print(output_dir)
model_name = "cnn_model.h5"

os.makedirs(output_dir, exist_ok=True)

def preparedata(file):
        
        basename=os.path.basename(file).replace(".edf","")
        raw = mne.io.read_raw_edf(file)
        a=int(len(raw)/3750)
        b=np.zeros(a)
        c=b+1
        d=np.linspace(0,a*3750,a,endpoint=False,dtype=int)
        e=np.vstack((d,b,c)).T.astype(int)
        event_id = {'Sleep stage W': 1}
        epochs = pr.sleep_stage_epochs(raw=raw, events=e, event_id=event_id).load_data()
        x_train=eeg_power_band_shhs(epochs).transpose(0,2,1)
        y_train = (epochs.events[:, 2]-1)
        filename = os.path.join(output_dir, basename+'.npz')
        save_dict = {
            "x": x_train, 
            "y": y_train, 
                }
        np.savez(filename, **save_dict)
        pass                    

preparedata(filename)

scorefiles = sorted(glob(os.path.join(output_dir, "*.npz")))
for files in scorefiles:
    files2=files.split()
    basename=os.path.basename(''.join(files)).replace(".npz","")
    score_dict = {k: np.load(k) for k in files2}
    model = get_model_cnn()
    model.load_weights(model_name)

    preds = []


    for record in tqdm(score_dict):
        all_rows = score_dict[record]['x']
        for batch_hyp in chunker(range(all_rows.shape[0])):


            X = all_rows[min(batch_hyp):max(batch_hyp)+1, ...]
            X = np.expand_dims(X, 0)
            X = rescale_array(X)

            Y_pred = model.predict(X)
            Y_pred = Y_pred.argmax(axis=-1).ravel().tolist()
            preds += Y_pred

    score_result = basename+'_score.csv'
    
    output_file = open(os.path.join(output_dir, score_result), 'w')
    output_file.write("stages\n")
    for y in preds:    
        output_file.write("{}\n".format(y))
    output_file.close()