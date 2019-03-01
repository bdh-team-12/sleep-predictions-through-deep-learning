import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import psd_array_welch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

ALL_SUBJECTS = list(range(20))

mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

# With ALL data
all_files_first_night = fetch_data(subjects=ALL_SUBJECTS, recording=[1])
all_raw_edf = []

for file in all_files_first_night:
    raw = mne.io.read_raw_edf(file[0])
    annot = mne.read_annotations(file[1])

    raw.set_annotations(annot, emit_warning=False)
    raw.set_channel_types(mapping)

    all_raw_edf.append(raw)

annotation_desc_2_event_id = {'Sleep stage W': 1,
                              'Sleep stage 1': 2,
                              'Sleep stage 2': 3,
                              'Sleep stage 3': 4,
                              'Sleep stage 4': 4,
                              'Sleep stage R': 5}

all_events = []
for raw_sample in all_raw_edf:
    events, _ = mne.events_from_annotations(raw_sample, event_id=annotation_desc_2_event_id, chunk_duration=30.)
    all_events.append(events)

event_id = {'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5}

all_epochs = []
for (raw, events) in zip(all_raw_edf, all_events):
    t_max = 30. - 1. / raw.info['sfreq']
    epochs = mne.Epochs(raw=raw, events=events, event_id=event_id,
                        tmin=0, tmax=t_max, baseline=None)
    all_epochs.append(epochs)

def eeg_power_band(epochs):
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
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}

    EEG_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]

    sfreq = epochs.info['sfreq']
    data = epochs.load_data().pick_channels(EEG_CHANNELS).get_data()
    psds, freqs = psd_array_welch(data, sfreq, fmin=0.5, fmax=30.,
                                  n_fft=512, n_overlap=256)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for _, (fmin, fmax) in FREQ_BANDS.items():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)


all_data = []
all_target = []
for epochs in all_epochs:
    data = eeg_power_band(epochs)
    all_data.append(data)

    events = epochs.events
    targets = events[:, 2]
    all_target.append(targets)

test_size = 5/20

X_train, X_test, y_train, y_test = train_test_split(all_data, all_target, test_size=0.4, random_state=0)

rfc = RandomForestClassifier(n_estimators=50, random_state=42, warm_start=True)

for (X, y) in zip(X_train, y_train):
    rfc.fit(X, y)
    rfc.n_estimators += 50

all_acc = []
for (Xt, yt) in zip(X_test, y_test):
    yp = rfc.predict(Xt)
    acc = accuracy_score(yt, yp)
    all_acc.append(acc)

print("Accuracy scores for cross validation: {}".format(all_acc))
