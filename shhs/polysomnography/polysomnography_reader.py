import mne
from mne.time_frequency import psd_array_welch
from shhs.parser import xml_nsrr as xn

# To make mne.Annotations, we'll need 3 arrays of equal length
# 1: onset
# 2: duration
# 3: description
# then we can call the Annotation initializer with orig_time=None
"""
TODO: Delete this example
Example:
onset, duration, description = _read_annotations_edf(fname)
        onset = np.array(onset, dtype=float)
        duration = np.array(duration, dtype=float)
        annotations = Annotations(onset=onset, duration=duration,
                                  description=description,
                                  orig_time=None)
"""


def annotations_from_nsrr_xml(xml_file_path):
    stages_elements = xn.parse_nsrr_sleep_stages(xml_file_path)

    stage = [elem.find('EventConcept').text for elem in stages_elements]
    onset = [elem.find('Start').text for elem in stages_elements]
    duration = [elem.find('Duration').text for elem in stages_elements]

    return stage, onset, duration


if __name__ == "__main__":
    file_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml"
    stage, onset, duration = annotations_from_nsrr_xml(file_path)
