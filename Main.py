import matplotlib.pyplot as plt
from shhs.polysomnography import polysomnography_reader as pr
import mne

if __name__ == "__main__":
    edf_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/edfs/shhs1/shhs1-200001.edf"
    ann_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml"

    raw = pr.annotated_raw_edf(edf_file_path=edf_path, annotations_file_path=ann_path)

    events, event_id = pr.sleep_stage_events(raw)
    epochs = pr.sleep_stage_epochs(raw=raw, events=events, event_id=event_id).load_data()

    # plot events
    ratio = 2.5
    h = 3
    w = h * ratio

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

    # Get spectral density plot
    plot_epochs = epochs.pick(['EEG'])

    for stage, color in zip(event_id.keys(), stage_colors[0:5]):
        plot_epochs[stage].plot_psd(area_mode=None, color=color, ax=ax2,
                                    fmin=0.1, fmax=20., show=False)

    ax2.set_title("Spectral Density")
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('uV^2/hz (dB)')

    fig.tight_layout()
    fig.show()

    print("done")
