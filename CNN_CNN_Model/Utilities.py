import h5py
import numpy as np
import random
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import psd_array_welch

WINDOW_SIZE = 300

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
    EEG_CHANNELS = ["EEG"]

    sfreq = epochs.info['sfreq']
    data = epochs.load_data().pick_channels(EEG_CHANNELS).get_data()*100000
    psds, freqs = psd_array_welch(data, sfreq, fmin=0.5, fmax=50.,
                                  n_fft=512, n_overlap=256)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)
    
    return psds*5

def rescale_array(X):
    X = X*5
    X = np.clip(X, -5, 5)
    return X


def aug_X(X):
    scale = 1 + np.random.uniform(-0.1, 0.1)
    offset = np.random.uniform(-0.1, 0.1)
    noise = np.random.normal(scale=0.05, size=X.shape)
    X = scale * X + offset + noise
    return X

def gen(dict_files, aug=False):
    while True:
        record_name = random.choice(list(dict_files.keys()))
        batch_data = dict_files[record_name]
        all_rows = batch_data['x']

        for i in range(10):
            start_index = random.choice(range(all_rows.shape[0]-WINDOW_SIZE))

            X = all_rows[start_index:start_index+WINDOW_SIZE, ...]
            Y = batch_data['y'][start_index:start_index+WINDOW_SIZE]

            X = np.expand_dims(X, 0)
            Y = np.expand_dims(Y, -1)
            Y = np.expand_dims(Y, 0)

            if aug:
                X = aug_X(X)
            X = rescale_array(X)

            yield X, Y


def chunker(seq, size=WINDOW_SIZE):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))