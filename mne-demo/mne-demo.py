from mne.io import read_raw_edf
from mne import events_from_annotations
from mne import read_annotations


def import_edf(file_path):
    ann = read_annotations(file_path)
    raw = read_raw_edf(file_path, preload=True, verbose=True)
    events, event_ids = events_from_annotations(raw, verbose=True)
    return ann, raw, events


if __name__ == "__main__":
    file_path = "../edf-samples/shhs_sample/0000.edf"
    (annotations, raw, events) = import_edf(file_path)
