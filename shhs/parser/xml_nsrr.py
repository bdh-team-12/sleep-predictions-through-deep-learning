from xml.etree import ElementTree


def parse_nsrr_annotations(file_path):
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    return root


def parse_nsrr_scored_events(file_path):
    root = parse_nsrr_annotations(file_path)
    scored_events = root.find('ScoredEvents').getchildren()
    return scored_events


def parse_nsrr_sleep_stages(file_path):
    events = parse_nsrr_scored_events(file_path)
    sleep_stages = [event for event in events if event.find('EventType').text == 'Stages|Stages']
    return sleep_stages


if __name__ == "__main__":
    file_path = "./dev/data/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml"
    result = parse_nsrr_sleep_stages(file_path)
