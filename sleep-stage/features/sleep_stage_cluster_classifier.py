from sklearn.model_selection import train_test_split

import shhs.polysomnography.polysomnography_reader as ps


def main():
    edf_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/edfs/shhs1"
    ann_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/annotations-events-nsrr/shhs1"
    epochs = ps.load_shhs_epoch_data(edf_path, ann_path, limit=150)
    train_test, validation = train_test_split(epochs, train_size=0.2)
    train, test = train_test_split(train_test, train_size=0.2)

    train_loader = generate_feature_classifier_data_loader(train, feature_window=10, batch_size=10, trim_zeros=True)
    test_loader = generate_feature_classifier_data_loader(test, feature_window=10, batch_size=10, trim_zeros=False)
    validation_loader = generate_feature_classifier_data_loader(validation,
                                                                feature_window=10,
                                                                batch_size=10,
                                                                trim_zeros=False)
    enc = train_feature_classifier(feature_width=10,
                                   train_data_loader=train_loader,
                                   test_data_loader=test_loader,
                                   num_epochs=10)
    plot_results(enc, validation_loader)
    print("Done?")


if __name__ == "__main__":
    main()