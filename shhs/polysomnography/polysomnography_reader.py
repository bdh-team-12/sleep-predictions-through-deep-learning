import mne
from shhs.parser import xml_nsrr as xn
import numpy as np
import os
import re


def load_shhs_raw_annotated_edfs(edf_path, annotations_path, limit=-1):
    # Find edf file paths
    edf_file_paths = []
    for file in os.listdir(edf_path):
        if file.endswith(".edf"):
            edf_file_paths.append(file)

    # Find annotation file paths
    annotation_file_paths = []
    for file in os.listdir(annotations_path):
        if file.endswith(".xml"):
            annotation_file_paths.append(file)

    # Match edf paths to annotation paths and generate annotated edf objects
    annotated_edfs = []
    for ann in annotation_file_paths:
        matches = [edf for edf in edf_file_paths if re.split("-|\.", ann)[1] == re.split("-|\.", edf)[1]]
        if len(matches) == 0:
            continue

        edf = matches[0]
        annotated_edfs.append(annotated_raw_edf(edf_file_path=os.path.join(edf_path, edf),
                                                annotations_file_path=os.path.join(annotations_path, ann)))

        if len(annotated_edfs) == limit:
            break

    return annotated_edfs


def load_shhs_epoch_data(edf_path, annotations_path, limit=-1):
    raw_edfs = load_shhs_raw_annotated_edfs(edf_path=edf_path,
                                            annotations_path=annotations_path,
                                            limit=limit)

    events_and_id = [(raw, sleep_stage_events(raw)) for raw in raw_edfs]
    epochs = [sleep_stage_epochs(raw, event_info[0], event_info[1]) for (raw, event_info) in events_and_id]
    return epochs


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
        event_id = {'Wake|0': 0,
                    'Stage 1 sleep|1': 1,
                    'Stage 2 sleep|2': 2,
                    'Stage 3 sleep|3': 3,
                    'REM sleep|5': 4}

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


def eeg_power_band(epochs, channels):
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

    channels: List<String>
        The list of channels to convert into power band features

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

    sampling_freq = epochs.info['sfreq']
    data = epochs.load_data().pick_channels(channels).get_data()
    psds, freqs = mne.time_frequency.psd_array_welch(data, sampling_freq, fmin=0.5, fmax=30.,
                                                     n_fft=512, n_overlap=256)
    # psds: [# epochs x 1 x # freq bins]
    # Each element in psds y = [x, :, :] is a  [1 x #_freq_bins] array containing the number of times each frequency
    # was observed.
    #
    # freqs: [# freq bins x 1], where each value is the median for the frequency bin

    # Normalize the PSDs so that each element y = [x, :, :] is normalized to sum to 1
    psds /= np.sum(psds, axis=-1, keepdims=True)

    return psds
