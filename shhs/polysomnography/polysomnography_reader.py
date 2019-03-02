import mne
from shhs.parser import xml_nsrr as xn
import numpy as np
import matplotlib.pyplot as plt


def annotation_components_from_nsrr_xml(xml_file_path):
    stages_elements = xn.parse_nsrr_sleep_stages(xml_file_path)

    stage = [elem.find('EventConcept').text for elem in stages_elements]
    onset = [elem.find('Start').text for elem in stages_elements]
    duration = [elem.find('Duration').text for elem in stages_elements]

    onset = np.array(onset, dtype=float)
    duration = np.array(duration, dtype=float)

    return stage, onset, duration


def annotations_from_nsrr_xml(xml_file_path):
    stage, onset, duration = annotation_components_from_nsrr_xml(xml_file_path)
    annotations = mne.Annotations(onset=onset, duration=duration,
                                  description=stage,
                                  orig_time=None)

    return annotations


if __name__ == "__main__":
    edf_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/edfs/shhs1/shhs1-200001.edf"
    file_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml"

    raw = mne.io.read_raw_edf(edf_path)
    annotations = annotations_from_nsrr_xml(file_path)
    raw.set_annotations(annotations)

    annotation_desc_2_event_id = {'Wake|0': 1,
                                  'Stage 1 sleep|1': 2,
                                  'Stage 2 sleep|2': 3,
                                  'Stage 3 sleep|3': 4,
                                  'REM sleep|5': 5}

    events, _ = mne.events_from_annotations(raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3': 4,
                'Sleep stage R': 5}

    # plot events
    fig = mne.viz.plot_events(events, event_id=event_id,
                              sfreq=raw.info['sfreq'], show=False)

    # keep the color-code for further plotting
    stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig.tight_layout()
    fig.show()

    # Get spectral density plot
    tmax = 30. - 1. / raw.info['sfreq']  # tmax in included

    epochs = mne.Epochs(raw=raw, events=events, event_id=event_id, tmin=0., tmax=tmax, baseline=None).load_data()

    plot_epochs = epochs.pick(['EEG'])
    fig, (ax) = plt.subplots(ncols=1)

    for stage, color in zip(event_id.keys(), stage_colors[0:5]):
        plot_epochs[stage].plot_psd(area_mode=None, color=color, ax=ax,
                                    fmin=0.1, fmax=20., show=False)

    ax.set_title("Spectral Density of Sleep Stages")
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('uV^2/hz (dB)')
    fig.legend(list(event_id.keys()))

    fig.tight_layout()
    fig.show()

    print("done")
