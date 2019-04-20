import torch
import torch.utils.data
import torch.optim
import torch.nn as nn

import shhs.polysomnography.polysomnography_reader as ps

# Takes sleep stage data and encodes it into more salient features
class SleepStageAutoEncoder(nn.Module):
    def __init__(self, n_input):
        super(SleepStageAutoEncoder, self).__init__()
        n_hidden_1 = n_input*0.75
        n_hidden_2 = n_hidden_1 * 0.5
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
    windows = [event_sample[i:i+feature_window]
               for event_sample in raw_events
               for i in range(0,
                              len(event_sample),
                              feature_window)]
    windows = [window for window in windows if len(window) == feature_window]

    # Convert event arrays into torch tensor dataset
    tens = torch.tensor(windows)
    dataset = torch.utils.data.TensorDataset(tens)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def train_feature_classifier(feature_width, dataloader, num_epochs=50):
    encoder = SleepStageAutoEncoder(n_input=feature_width)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(encoder.parameters())

    for epoch in range(num_epochs):
        for data in dataloader:
            train_x, train_y = data
            # ===================forward=====================
            output = encoder(train_x)
            loss = criterion(output, train_x)
            MSE_loss = nn.MSELoss()(output, train_x)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data[0], MSE_loss.data[0]))

    torch.save(encoder.state_dict(), './autoencoder.pth')
    return encoder


def main():
    edf_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/edfs/shhs1"
    ann_path = "/Users/blakemacnair/dev/data/shhs/polysomnography/annotations-events-nsrr/shhs1"
    epochs = ps.load_shhs_epoch_data(edf_path, ann_path)


if __name__ == "__name__":
    main()
