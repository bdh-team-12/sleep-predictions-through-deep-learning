import torch.nn as nn
import torch.optim
import torch.utils.data
from sklearn.model_selection import train_test_split

import shhs.polysomnography.polysomnography_reader as ps
from plots import *
from utils import *
import torch.nn.functional as F

import pandas as pd

from sleep_stage.features.sleep_stage_auto_encoder import (load_feature_classifier, SleepStageAutoEncoder)


class SleepStageClusterClassifier(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(SleepStageClusterClassifier, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=40, kernel_size=5, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(40 * 268, 1000)
        self.fc2 = nn.Linear(1000, dim_output)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, 40 * 268)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x


def generate_event_cluster_data_loader(data, batch_size=10):
    # Convert arrays into torch tensor [identity] dataset
    seq = [x[0] for x in data]
    cluster = [x[1] for x in data]

    seq_tens = torch.tensor(seq).float()
    cluster_tens = torch.tensor(cluster).long()
    dataset = torch.utils.data.TensorDataset(seq_tens.unsqueeze(1), cluster_tens)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def train_cluster_classifier(max_sequence_length, cluster_count, train_data_loader, test_data_loader, num_epochs=50,
                             criterion=nn.CrossEntropyLoss()):
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    model = SleepStageClusterClassifier(dim_input=max_sequence_length, dim_output=cluster_count)
    optimizer = torch.optim.Adam(model.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, device, train_data_loader, criterion, optimizer, epoch,
                                           print_freq=100)
        valid_loss, valid_accuracy, valid_results = evaluate(model, device, test_data_loader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

    plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)
    return model


def plot_results(model, test_loader, classes):
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    test_loss, test_accuracy, test_results = evaluate(model, device, test_loader, criterion)

    plot_confusion_matrix(test_results, classes)


def main():
    edf_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/edfs/shhs1"
    ann_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/annotations-events-nsrr/shhs1"
    clusters_path = "/Users/blakemacnair/dev/data/ClusterID.csv"
    enc_path = "./../features/autoencoder.pth"
    sample_limit = 200

    # Read in cluster assignments
    data = pd.read_csv(clusters_path, ',')
    ids = data['patientID'].array
    clusters = data['clusterID'].array

    epochs = ps.load_shhs_epoch_data(edf_path, ann_path, limit=sample_limit)

    ev = [ep.events[:, 2] for ep in epochs]
    pad_max = max([len(x) for x in ev])

    # Pad sequences to match longest event sequence
    events = [numpy.append(e, numpy.array([0] * (pad_max - len(e)))) for e in ev]

    # Sample cluster assignments
    max_cluster_id = 5
    clusters = numpy.random.randint(low=0, high=max_cluster_id, size=len(events))

    # Combine event sequences with their cluster assignments
    dataset = [(events[i], clusters[i]) for i in range(len(events))]

    # Prepare training and testing and validation data
    train_test, validation = train_test_split(dataset, train_size=0.5)
    train, test = train_test_split(train_test, train_size=0.4)

    # Create data loaders
    train_loader = generate_event_cluster_data_loader(train, batch_size=20)
    test_loader = generate_event_cluster_data_loader(test, batch_size=20)
    validation_loader = generate_event_cluster_data_loader(validation, batch_size=20)

    model = train_cluster_classifier(max_sequence_length=pad_max, cluster_count=max_cluster_id+1,
                                     train_data_loader=train_loader, test_data_loader=test_loader,
                                     num_epochs=20)

    plot_results(model, validation_loader, list(range(max_cluster_id)))


if __name__ == "__main__":
    main()
