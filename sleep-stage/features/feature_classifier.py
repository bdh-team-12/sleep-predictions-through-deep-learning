import torch
import torch.nn as nn
import torch.nn.functional as functional



# Takes a time-sequence of sleep stages and returns probabilities of fitting into different disease risk groups
class SleepStageFeatureClassifier(torch.nn.Module):
    def __init__(self):
        super(SleepStageFeatureClassifier, self).__init__()

    def forward(self, x):
        return x


class SleepStageFeaturesCNN(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(SleepStageFeaturesCNN, self).__init__()

        self.hidden1 = nn.Linear(n_input, n_hidden * 2)
        self.hidden2 = nn.Linear(n_hidden * 2, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = functional.leaky_relu(self.hidden1(x))
        x = functional.leaky_relu(self.hidden2(x))
        x = functional.leaky_relu(self.out(x))
        return x
