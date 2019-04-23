from sklearn.model_selection import train_test_split

import shhs.polysomnography.polysomnography_reader as ps
from sleep_stage.features.feature_classifier import load_feature_classifier


def main():
    edf_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/edfs/shhs1"
    ann_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/annotations-events-nsrr/shhs1"
    enc_path = "./../features/autoencoder.pth"

    enc = load_feature_classifier(enc_path)

    print("Done?")


if __name__ == "__main__":
    main()