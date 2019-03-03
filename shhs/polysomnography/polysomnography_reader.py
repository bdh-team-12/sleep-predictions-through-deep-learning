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
