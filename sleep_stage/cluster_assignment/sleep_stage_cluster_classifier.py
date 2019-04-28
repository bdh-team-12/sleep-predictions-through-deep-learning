import torch.nn as nn
import torch.optim
import torch.utils.data
from sklearn.model_selection import train_test_split

import shhs.polysomnography.polysomnography_reader as ps
from plots import *
from utils import *

from sleep_stage.features.sleep_stage_auto_encoder import (load_feature_classifier, SleepStageAutoEncoder)

class SleepStageClusterClassifier(nn.Module):
    def __init__(self, dim_input):
        super(SleepStageClusterClassifier, self).__init__()
        self.dim_input = dim_input

        self.sequence = nn.Sequential(
            nn.Linear(dim_input, 100),
            nn.LeakyReLU(True),
            nn.GRU(input_size=100, hidden_size=50, num_layers=1, batch_first=True, dropout=0),
            nn.Sigmoid(),
            nn.GRU(input_size=50, hidden_size=32, num_layers=2, batch_first=True, dropout=0.2),
            nn.Sigmoid(),
            nn.GRU(input_size=32, hidden_size=16, num_layers=3, batch_first=True, dropout=0.4),
            nn.Sigmoid()
        )

        self.transformer = nn.Sequential(
            nn.Linear(in_features=16, out_features=2),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        x = self.sequence(x)
        x = self.transformer(x[:, -1, :])
        return x


def main():
    edf_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/edfs/shhs1"
    ann_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/annotations-events-nsrr/shhs1"
    enc_path = "./../features/autoencoder.pth"
    sample_limit = 10

    epochs = ps.load_shhs_epoch_data(edf_path, ann_path, limit=sample_limit)
    train_test, validation = train_test_split(epochs, train_size=0.2)
    train, test = train_test_split(train_test, train_size=0.2)

    enc = load_feature_classifier(enc_path)

    print("Done")


if __name__ == "__main__":
    main()
