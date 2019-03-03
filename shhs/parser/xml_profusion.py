from xml.etree import ElementTree


def parse_profusion_annotations(file_path):
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    return root


# Source for format https://github.com/nsrr/edf-editor-translator/wiki/Compumedics-Annotation-Format
def parse_profusion_sleep_stages(file_path):
    root = parse_profusion_annotations(file_path)
    sleep_stages = root.find('SleepStages').getchildren()
    return sleep_stages
