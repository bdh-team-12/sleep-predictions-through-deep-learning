import torch
import torch.utils.data
import torch.optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy
import time

from utils import *
from plots import *

import shhs.polysomnography.polysomnography_reader as ps


# Takes sleep stage data and encodes it into more salient features
class SleepStageAutoEncoder(nn.Module):
    def __init__(self, n_input):
        super(SleepStageAutoEncoder, self).__init__()
        n_hidden_1 = round(n_input * 0.75)
        n_hidden_2 = round(n_hidden_1 * 0.5)
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden_1),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_1),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_input))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def generate_feature_classifier_data_loader(epoch_data, feature_window, batch_size=10):
    raw_events = [epoch.events[:, 2] for epoch in epoch_data]

    # Split event data into windows to identify features
    windows = [event_sample[i:i + feature_window]
               for event_sample in raw_events
               for i in range(0,
                              len(event_sample),
                              feature_window)]
    windows = [window for window in windows if len(window) == feature_window]

    # Convert event arrays into torch tensor [identity] dataset
    tens = torch.tensor(windows)
    dataset = torch.utils.data.TensorDataset(tens.float(), tens.float())

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def train_feature_classifier(feature_width, train_data_loader, test_data_loader, num_epochs=50):
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    encoder = SleepStageAutoEncoder(n_input=feature_width)
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(encoder.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    criterion.to(device)

    encoder.train()

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(encoder, device, train_data_loader, criterion, optimizer, epoch, print_freq=100)
        valid_loss, valid_accuracy, valid_results = evaluate(encoder, device, test_data_loader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

    plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)
    torch.save(encoder.state_dict(), './autoencoder.pth')
    return encoder


def main():
    edf_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/edfs/shhs1"
    ann_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/annotations-events-nsrr/shhs1"
    epochs = ps.load_shhs_epoch_data(edf_path, ann_path, limit=50)
    train, test = train_test_split(epochs, train_size=0.2)
    train_loader = generate_feature_classifier_data_loader(train, feature_window=10, batch_size=10)
    test_loader = generate_feature_classifier_data_loader(test, feature_window=10, batch_size=10)
    enc = train_feature_classifier(feature_width=10,
                                   train_data_loader=train_loader,
                                   test_data_loader=test_loader,
                                   num_epochs=10)
    print("Done?")


if __name__ == "__main__":
    main()
