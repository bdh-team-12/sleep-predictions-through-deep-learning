"""
Created on Sat Apr  6 12:44:30 2019

@author: CRNZ
"""
    
import numpy as np
import pandas as pd
import scipy.signal as ssignal
import os
import matplotlib.pyplot as plt
"""this is from Blake's code"""      
from shhs.polysomnography import polysomnography_reader as pr
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import psd_array_welch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer    
from sklearn.preprocessing import StandardScaler
from utils import train, evaluate
from plots import plot_learning_curves, plot_confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset
import torch.nn.functional as F
from torch.autograd import Variable
import glob
import math

def eeg_power_band_shhs(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    '''0.75-4.50 Hz (Delta), 4.75-7.75 (Theta), 8.00-12.25 (Alpha), 12.50-15.00 (Sigma), 15.25-24.75 (Beta), 25.00-34.75 (Gamma 1), and 35.00-44.75 (Gamma 2) The '''
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30],
                  "Gamma":[30,50]}
    '''FREQ_BANDS = {"delta1": [0.5, 2.5],
                  "delta2": [2.5, 4.5],
                  "theta1": [4.5, 6.5],
                  "theta2": [6.5, 8.5],                  
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30],
                  "Gamma":[30,50]}'''
    EEG_CHANNELS = ["EEG(sec)", "EEG"]

    sfreq = epochs.info['sfreq']
    data = epochs.load_data().pick_channels(EEG_CHANNELS).get_data()
    psds, freqs = psd_array_welch(data, sfreq, fmin=0.5, fmax=50.,
                                  n_fft=512, n_overlap=256)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for _, (fmin, fmax) in FREQ_BANDS.items():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)    

def add_pre_post(data):
    '''EEG relative power band feature extraction'''
    df=pd.DataFrame(data)
    df2=df.shift(1).fillna(0)
    df3=df.shift(-1).fillna(0)
    df4=df.shift(2).fillna(0)
    df5=df.shift(-2).fillna(0)
    df_diff1=df.subtract(df2, fill_value=0)
    df_diff2=df.subtract(df3, fill_value=0)    
    dffinal=pd.concat([df, df2,df3], axis=1)
    return dffinal.values
'''read data'''

shhs_base_dir = 'D:/Documents/GaTech/CSE 6250 Big data for Health/Term project/Ruby for download/shhs/polysomnography'

def preparedata(sampling_rate,proportions = (0.5, 0.2, 0.3)):
    preprocessed_names = glob.glob(os.path.join(
    shhs_base_dir, 'edfs','shhs1', '*.edf'))
    nb_patients=math.ceil(len(preprocessed_names)*sampling_rate)
    r = np.arange(len(preprocessed_names))
    np.random.shuffle(r)
    preprocessed_names = [preprocessed_names[i] for i in r]
    preprocessed_names = preprocessed_names[:nb_patients]
    
    n_train = int(proportions[0]*len(preprocessed_names))
    print('n_train: ', n_train)
    n_valid = int(proportions[1]*len(preprocessed_names))
    print('n_valid: ', n_valid)
    names_train = preprocessed_names[0:n_train]
    names_valid = preprocessed_names[n_train:n_train+n_valid]
    names_test = preprocessed_names[n_train+n_valid:]

    for file in names_train:
        '''print('filename: ', os.path.basename(file))'''
        basename=os.path.basename(file).replace(".edf","")
        '''print('filename: ', basename)'''
        
        annotation=os.path.join(shhs_base_dir, 'annotations-events-nsrr','shhs1', basename+'-nsrr.xml')
        '''print('annotation: ', annotation)'''
        exists=os.path.isfile(annotation)
        if exists:
            raw = mne.io.read_raw_edf(file)
            annotations = pr.nsrr_sleep_stage_annotations(annotation)
            raw.set_annotations(annotations)
            events, _ = pr.sleep_stage_events(raw)
            event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3': 4,
                'Sleep stage R': 5}
            if len(set(events[:,2]))==5:
                epochs = pr.sleep_stage_epochs(raw=raw, events=events, event_id=event_id).load_data()
                x_train=eeg_power_band_shhs(epochs)
                x_train2=add_pre_post(x_train)
                y_train = epochs.events[:, 2]-1
                if x_train.shape[1]==12:                 
                    try:
                        _ = X_train.shape
                        X_train=np.concatenate((X_train, x_train2))
                        Y_train=np.concatenate((Y_train, y_train))
                    except NameError:
                        X_train=x_train2
                        Y_train=y_train
                
    for file in names_valid:
        '''print('filename: ', os.path.basename(file))'''
        basename=os.path.basename(file).replace(".edf","")
        '''print('filename: ', basename)'''
        
        annotation=os.path.join(shhs_base_dir, 'annotations-events-nsrr','shhs1', basename+'-nsrr.xml')
        '''print('annotation: ', annotation)'''
        exists=os.path.isfile(annotation)
        if exists:
            raw = mne.io.read_raw_edf(file)
            annotations = pr.nsrr_sleep_stage_annotations(annotation)
            raw.set_annotations(annotations)
            events, _ = pr.sleep_stage_events(raw)
            event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3': 4,
                'Sleep stage R': 5}
            if len(set(events[:,2]))==5:
                epochs = pr.sleep_stage_epochs(raw=raw, events=events, event_id=event_id).load_data()
                x_valid=eeg_power_band_shhs(epochs)
                x_valid2=add_pre_post(x_valid)
                y_valid = epochs.events[:, 2]-1
                if x_valid.shape[1]==12:                 
                    try:
                        _ = X_valid.shape
                        X_valid=np.concatenate((X_valid, x_valid2))
                        Y_valid=np.concatenate((Y_valid, y_valid))
                    except NameError:
                        X_valid=x_valid2
                        Y_valid=y_valid
                        
                
    for file in names_test:
        '''print('filename: ', os.path.basename(file))'''
        basename=os.path.basename(file).replace(".edf","")
        '''print('filename: ', basename)'''
        
        annotation=os.path.join(shhs_base_dir, 'annotations-events-nsrr','shhs1', basename+'-nsrr.xml')
        '''print('annotation: ', annotation)'''
        exists=os.path.isfile(annotation)
        if exists:
            raw = mne.io.read_raw_edf(file)
            annotations = pr.nsrr_sleep_stage_annotations(annotation)
            raw.set_annotations(annotations)
            events, _ = pr.sleep_stage_events(raw)
            event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3': 4,
                'Sleep stage R': 5}
            if len(set(events[:,2]))==5:
                epochs = pr.sleep_stage_epochs(raw=raw, events=events, event_id=event_id).load_data()
                x_test=eeg_power_band_shhs(epochs)
                x_test2=add_pre_post(x_test)
                y_test = epochs.events[:, 2]-1
                if x_test.shape[1]==12:                 
                    try:
                        _ = X_test.shape
                        X_test=np.concatenate((X_test, x_test2))
                        Y_test=np.concatenate((Y_test, y_test))
                    except NameError:
                        X_test=x_test2
                        Y_test=y_test
                
    return X_train,Y_train,X_valid,Y_valid,X_test,Y_test

X_train,Y_train,X_valid,Y_valid,X_test,Y_test=preparedata(0.02)

scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_valid=scaler.transform(X_valid)
X_test=scaler.transform(X_test)

"""
: count unique values
unique, counts = np.unique(y_train, return_counts=True)
np.asarray((unique, counts)).T
"""
MODEL_TYPE = 'MLP'  # TODO: Change this to 'MLP', 'CNN', or 'RNN' according to your task
NUM_EPOCHS = 15
BATCH_SIZE = 200
USE_CUDA = False  # Set 'True' if you want to use GPU
NUM_WORKERS = 0
def load_dataset(data,labels, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	# TODO: Read a csv file from path.
	# TODO: Please refer to the header of the file to locate X and y.
	# TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# TODO: Remove the header of CSV file of course.
	# TODO: Do Not change the order of rows.
	# TODO: You can use Pandas if you want to.

    
	if model_type == 'MLP':
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')), torch.from_numpy(labels.astype('long')).view(-1).long())
	elif model_type == 'CNN':
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(1), torch.from_numpy(labels.astype('long')).long())
	elif model_type == 'RNN':
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(2), torch.from_numpy(labels.astype('long')).long())
	elif model_type == 'LSTM':
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(2), torch.from_numpy(labels.astype('long')).long())
        
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset

train_dataset = load_dataset(X_train,Y_train, MODEL_TYPE)
valid_dataset = load_dataset(X_valid,Y_valid, MODEL_TYPE)
test_dataset = load_dataset(X_test,Y_test, MODEL_TYPE)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

'''
train_dataset = load_dataset(X_train_new,y_train, MODEL_TYPE)
valid_dataset = load_dataset(X_valid_new,y_valid, MODEL_TYPE)
test_dataset = load_dataset(X_test_new,y_test, MODEL_TYPE)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
'''

PATH_OUTPUT = "../output/sleep/"
os.makedirs(PATH_OUTPUT, exist_ok=True)
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
 
if MODEL_TYPE == 'MLP':
	model = MyMLP()
	save_file = 'MyMLP.pth'
elif MODEL_TYPE == 'CNN':
	model = MyCNN()
	save_file = 'MyCNN.pth'
elif MODEL_TYPE == 'RNN':
	model = MyRNN()
	save_file = 'MyRNN.pth'
elif MODEL_TYPE == 'LSTM':
	model=LSTM(1,20,5)
	model.initWeight
	model.initHidden(30)
	save_file = 'MyLSTM.pth'
    
else:
	raise AssertionError("Wrong Model Type!")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)

best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []

for epoch in range(NUM_EPOCHS):
	train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
	valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

	train_losses.append(train_loss)
	valid_losses.append(valid_loss)

	train_accuracies.append(train_accuracy)
	valid_accuracies.append(valid_accuracy)

	is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
	if is_best:
		best_val_acc = valid_accuracy
		torch.save(model, os.path.join(PATH_OUTPUT, save_file))
plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)        
best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))        


test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader, criterion)
class_names = ['Wake', 'Stage 1', 'Stage 2', 'Stage 3', 'R']
plot_confusion_matrix(test_results, class_names)
acc_model=accuracy_score(np.asarray(test_results)[:,1],Y_test)
acc_model
f1score_model=f1_score(np.asarray(test_results)[:,1],Y_test,average="weighted")

rfc=RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, Y_train)
Y_pred=rfc.predict(X_train)
YP_test=rfc.predict(X_test)
acc=accuracy_score(YP_test,Y_test)
acc
plot_confusion_matrix(list(map(lambda x,y:(x,y),Y_test,YP_test)), class_names)
