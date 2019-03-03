import mne
from shhs.parser import xml_nsrr as xn
import numpy as np
import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    edf_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/edfs/shhs1/shhs1-200001.edf"
    file_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml"

    raw = mne.io.read_raw_edf(edf_path)
    annotations = nsrr_sleep_stage_annotations(file_path)
    raw.set_annotations(annotations)

    events, _ = sleep_stage_events(raw)

    event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3': 4,
                'Sleep stage R': 5}

    # plot events
    ratio = 2.5
    h = 3
    w = h * ratio
    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(w, h))

    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1)
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1)
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    mne.viz.plot_events(events, event_id=event_id, axes=ax1,
                        sfreq=raw.info['sfreq'], show=False)

    ax1.get_legend().remove()
    ax1.set_title("Sleep Stages over time")
    ax3.axis('off')

    fig.legend(list(event_id.keys()), loc=(0.4, 0.3), ncol=1, labelspacing=0.)

    # keep the color-code for further plotting
    stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig.tight_layout()
    fig.show()

    # Get spectral density plot
    epochs = sleep_stage_epochs(raw=raw, events=events, event_id=event_id).load_data()

    plot_epochs = epochs.pick(['EEG'])

    for stage, color in zip(event_id.keys(), stage_colors[0:5]):
        plot_epochs[stage].plot_psd(area_mode=None, color=color, ax=ax2,
                                    fmin=0.1, fmax=20., show=False)

    ax2.set_title("Spectral Density")
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('uV^2/hz (dB)')

    fig.tight_layout()
    fig.show()
    fig.savefig('sleep_stage_time_spectral.png')

    print("done")
