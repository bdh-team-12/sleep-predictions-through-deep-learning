import torch
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


def generate_data_loader(edf_data):
    break


def train_sleep_stage_auto_encoder(feature_width, dataloader, num_epochs=50):
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
    data = ps.load_shhs_raw_annotated_edfs(edf_path, ann_path)


if __name__ == "__name__":
    main()
