# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:40:33 2019

@author: CRNZ
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.autograd import Variable
class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=16, num_layers=3, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(in_features=16, out_features=5)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

class MyMLP(nn.Module):
    def __init__(self):
        super(MyMLP, self).__init__()
        self.hidden1 = nn.Linear(36, 32)
        self.hidden2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 5)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))        
        x = self.out(x)
        return x  
class MyCNN(nn.Module):
    ''' initial model
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)       
        self.fc1 = nn.Linear(in_features=16 * 41, out_features=128)
        self.fc2 = nn.Linear(128, 5)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 41)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        '''
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)       
        self.fc1 = nn.Linear(in_features=16 * 6, out_features=128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)        


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)
        return x

class MyCNN2(nn.Module):
    ''' initial model
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)       
        self.fc1 = nn.Linear(in_features=16 * 41, out_features=128)
        self.fc2 = nn.Linear(128, 5)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 41)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        '''
    def __init__(self):
        super(MyCNN2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=12, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(12, 16, 5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=16 * 6, out_features=128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)        


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)
        return x
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2, batch_size=100,gpu=False):
        super(LSTM, self).__init__()

        self._gpu        = gpu
        self.hidden_size = hidden_size
        self.num_layers  = num_layers


        self.lstm   = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.drop   = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden = self.initHidden(batch_size)
    def forward(self, inputs):
        # hidden = (h_t, c_t)
        '''print('input size',input.size())'''
        _, self.hidden = self.lstm(inputs)
        # extract the last hidden layer from h_t(n_layers, n_samples, hidden_size)
        htL = self.hidden[0][-1]
        htL = self.drop(htL)
        outputs = self.linear(htL)
        return outputs

    def initWeight(self, init_forget_bias=1):

        for name, params in self.named_parameters():

            if 'weight' in name:
                init.xavier_uniform(params)


            elif 'lstm.bias_ih_l' in name:
                b_ii, b_if, b_ig, b_i0 = params.chunk(4, 0)
                init.constant(b_if, init_forget_bias)
            elif 'lstm.bias_hh_l' in name:
                b_hi, b_hf, b_hg, b_h0 = params.chunk(4, 0)
                init.constant(b_hf, init_forget_bias)


            else:
                init.constant(params, 0)

    def initHidden(self, batch_size):
        if self._gpu == True:
            self.hidden = (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()))
        else:
            self.hidden = (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)),
                           Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)))

class LSTM2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size):
        super(LSTM2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)),
                Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x,self.hidden)
        htL = self.hidden[0][-1]
        output = self.proj(htL)
        result = F.sigmoid(output)
        return result
 
