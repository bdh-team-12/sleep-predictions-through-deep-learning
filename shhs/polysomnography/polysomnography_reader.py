import mne
from shhs.parser import xml_nsrr as xn
import numpy as np


def nsrr_sleep_stage_components(xml_file_path):
    stages_elements = xn.parse_nsrr_sleep_stages(xml_file_path)

    stage = [elem.find('EventConcept').text for elem in stages_elements]
    onset = [elem.find('Start').text for elem in stages_elements]
    duration = [elem.find('Duration').text for elem in stages_elements]

    onset = np.array(onset, dtype=float)
    duration = np.array(duration, dtype=float)

    return stage, onset, duration


def nsrr_sleep_stage_annotations(xml_file_path):
    stage, onset, duration = nsrr_sleep_stage_components(xml_file_path)
    annotations = mne.Annotations(onset=onset, duration=duration,
                                  description=stage,
                                  orig_time=None)

    return annotations


def sleep_stage_events(raw, event_id=None, chunk_duration=30.):
    if event_id is None:
        event_id = {'Wake|0': 1,
                    'Stage 1 sleep|1': 2,
                    'Stage 2 sleep|2': 3,
                    'Stage 3 sleep|3': 4,
                    'REM sleep|5': 5}

    events_out, event_ids_out = mne.events_from_annotations(raw,
                                                            event_id=event_id,
                                                            chunk_duration=chunk_duration)

    return events_out, event_ids_out


def sleep_stage_epochs(raw, events, event_id, tmin=None, tmax=None, baseline=None):
    tmin = tmin if tmin is not None else 0.
    tmax = tmax if tmax is not None else 30. - 1. / raw.info['sfreq']
    epochs = mne.Epochs(raw=raw, events=events, event_id=event_id,
                        tmin=tmin, tmax=tmax,
                        baseline=baseline)
    return epochs


def annotated_raw_edf(edf_file_path, annotations_file_path):
    raw = mne.io.read_raw_edf(edf_file_path)
    annotations = nsrr_sleep_stage_annotations(annotations_file_path)
    raw.set_annotations(annotations)
    return raw


def eeg_power_band(epochs, channel):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Function implementation taken from https://martinos.org/mne/dev/auto_tutorials/plot_sleep.html

    Excerpt taken from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2824445/ :
        Choose frequency bands. Common frequency bands are low delta 0.3-1 Hz, delta 1-4 Hz , theta 4-8 Hz,
        alpha 8-12 Hz, sigma 12-15 Hz, and beta 15-30 Hz. Depending on the focus of the study, it may be desirable to
        break these down into narrower bands, such as 1-2, 2-3, and 3-4 Hz. The true band limits will usually be
        different than these nominal values.

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

    sfreq = epochs.info['sfreq']
    data = epochs.load_data().pick_channels([channel]).get_data()
    psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq, fmin=0.5, fmax=30.,
                                                     n_fft=512, n_overlap=256)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for _, (fmin, fmax) in FREQ_BANDS.items():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)
