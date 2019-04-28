import torch.nn as nn
import torch.optim
import torch.utils.data
from sklearn.model_selection import train_test_split

import shhs.polysomnography.polysomnography_reader as ps
from plots import *
from utils import *


# Takes sleep stage data and encodes it into more abstract, salient features
class SleepStageAutoEncoder(nn.Module):
    def __init__(self, feature_window_len):
        super(SleepStageAutoEncoder, self).__init__()
        self.feature_window_len = feature_window_len
        n_hidden_1 = round(feature_window_len * 0.75)
        n_hidden_2 = round(n_hidden_1 * 0.5)
        self.encoder = nn.Sequential(
            nn.Linear(feature_window_len, n_hidden_1),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_1),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, feature_window_len))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        return x


def generate_feature_classifier_data_loader(epoch_data, feature_window, batch_size=10, trim_zeros=False):
    raw_events = [epoch.events[:, 2] for epoch in epoch_data]

    # Split event data into windows to identify features
    windows = [event_sample[i:i + feature_window]
               for event_sample in raw_events
               for i in range(0,
                              len(event_sample),
                              feature_window)]
    windows = [window for window in windows if len(window) == feature_window]

    if trim_zeros:
        windows = [window for window in windows if sum(window) > 0]

    # Convert event arrays into torch tensor [identity] dataset
    tens = torch.tensor(windows)
    dataset = torch.utils.data.TensorDataset(tens.float(), tens.float())

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def train_feature_classifier(feature_width, train_data_loader, test_data_loader, num_epochs=50, criterion=nn.MSELoss()):
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    encoder = SleepStageAutoEncoder(feature_window_len=feature_width)
    optimizer = torch.optim.Adam(encoder.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    criterion.to(device)

    encoder.train()

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(encoder, device, train_data_loader, criterion, optimizer, epoch,
                                           print_freq=100)
        valid_loss, valid_accuracy, valid_results = evaluate(encoder, device, test_data_loader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

    plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)
    return encoder


def new_feature_classifier(edf_path, annotations_path, sample_limit=-1):
    epochs = ps.load_shhs_epoch_data(edf_path, annotations_path, limit=sample_limit)
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

    enc.eval()
    return enc


def encode_sleep_stage_epoch(epoch, encoder):
    raw = epoch.events[:, 2]
    encode_sleep_stage_sequence(raw, encoder)


def encode_sleep_stage_sequence(sleep_stage_sequence, encoder):
    # Split event data into windows to identify features
    windows = [sleep_stage_sequence[i:i + encoder.feature_window_len]
               for i in range(0,
                              len(sleep_stage_sequence),
                              encoder.feature_window_len)]
    windows = [window for window in windows if len(window) == encoder.feature_window_len]

    # Encode sleep stage windows
    out = [encoder.encode(window) for window in windows]


def save_feature_classifier(model, path):
    torch.save(model, path)


def load_feature_classifier(path):
    model = torch.load(path)
    model.eval()
    return model


def plot_results(encoder, test_loader):
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    criterion.to(device)

    test_loss, test_accuracy, test_results = evaluate(encoder, device, test_loader, criterion)

    class_names = ['1', '2', '3', '4', '5']

    flattened_results = [(r[0][i], r[1][i]) for r in test_results for i in range(len(r[0]))]
    plot_confusion_matrix(flattened_results, class_names)


def main():
    edf_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/edfs/shhs1"
    ann_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/annotations-events-nsrr/shhs1"
    enc = new_feature_classifier(edf_path, ann_path, sample_limit=500)
    path = './autoencoder.pth'
    # save_feature_classifier(enc, path)
    # identical = load_feature_classifier(path)


if __name__ == "__main__":
    main()
