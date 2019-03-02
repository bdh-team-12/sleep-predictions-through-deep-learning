import numpy as np

import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import psd_array_welch

from statistics import mean

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ALL_SUBJECTS = list(range(20))

mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}

EEG_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]

event_id = {'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5}

annotation_desc_2_event_id = {'Sleep stage W': 1,
                              'Sleep stage 1': 2,
                              'Sleep stage 2': 3,
                              'Sleep stage 3': 4,
                              'Sleep stage 4': 4,
                              'Sleep stage R': 5}


def fetch_subjects_from_sleep_physionet():
    all_files_first_night = fetch_data(subjects=ALL_SUBJECTS, recording=[1])
    all_raw_edf = []

    for file in all_files_first_night:
        raw = mne.io.read_raw_edf(file[0])
        annot = mne.read_annotations(file[1])

        raw.set_annotations(annot, emit_warning=False)
        raw.set_channel_types(mapping)

        all_raw_edf.append(raw)

    return all_raw_edf


def extract_events_from_raw_edfs(raw_edfs):
    all_events = []
    for raw_sample in raw_edfs:
        events, _ = mne.events_from_annotations(raw_sample, event_id=annotation_desc_2_event_id, chunk_duration=30.)
        all_events.append(events)

    return all_events


def extract_epochs(raw_edfs, events):
    all_epochs = []
    for (raw, events) in zip(raw_edfs, events):
        t_max = 30. - 1. / raw.info['sfreq']
        epochs = mne.Epochs(raw=raw, events=events, event_id=event_id,
                            tmin=0, tmax=t_max, baseline=None)
        all_epochs.append(epochs)

    return all_epochs


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


def generate_power_band_data_from_epochs(epochs):
    all_data = []
    all_target = []
    for epoch in epochs:
        data = eeg_power_band(epoch)
        all_data.append(data)

        events = epoch.events
        targets = events[:, 2]
        all_target.append(targets)

    return all_data, all_target


def train_random_forest_classifier(X, y):
    rfc = RandomForestClassifier(n_estimators=50, random_state=42, warm_start=True)

    for (X_sample, y_sample) in zip(X, y):
        rfc.fit(X_sample, y_sample)
        rfc.n_estimators += 50

    return rfc


def test_random_forest_classifier(rfc, X, y):
    all_acc = []
    for (Xt, yt) in zip(X, y):
        yp = rfc.predict(Xt)
        acc = accuracy_score(yt, yp)
        all_acc.append(acc)

    return all_acc


if __name__ == "__main__":
    raw_edfs = fetch_subjects_from_sleep_physionet()
    events = extract_events_from_raw_edfs(raw_edfs)
    epochs = extract_epochs(raw_edfs, events)

    X, y = generate_power_band_data_from_epochs(epochs)

    test_size = 5 / 20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    rfc = train_random_forest_classifier(X_train, y_train)

    scores = test_random_forest_classifier(rfc, X_test, y_test)

    print("Accuracy scores for cross validation: {}".format(scores))
    print("Average accuracy: {}".format(mean(scores)))
