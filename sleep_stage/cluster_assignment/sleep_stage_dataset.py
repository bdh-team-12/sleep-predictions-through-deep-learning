from torch.utils.data import Dataset


class SleepStageSequenceDataset(Dataset):
    def __init__(self, seqs, labels, num_features):
        """
        Args:
            seqs (list): list of matrices whose i-th value is the i-th time step in the auto-encoded data for a
            subject, and the j-th column is the index for the j-th auto-encoded feature out of all possible features
            labels (list): list of subject cluster labels (int)
            num_features (int): number of total features available
        """

        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")

        self.labels = labels
        self.seqs = seqs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # returns will be wrapped as List of Tensor(s) by DataLoader
        return self.seqs[index], self.labels[index]
