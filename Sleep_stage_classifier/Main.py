# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:36:31 2019

@author: CRNZ
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

from load_dataset import load_dataset
from utils import train, evaluate
from plots import plot_learning_curves, plot_confusion_matrix
from Models import MyMLP, MyCNN, MyRNN, LSTM

from sklearn.metrics import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer    
from sklearn.preprocessing import StandardScaler
import pickle

PATH_OUTPUT = "../output/model/"
os.makedirs(PATH_OUTPUT, exist_ok=True)


# Set a correct path to the seizure data file you downloaded
PATH_TRAIN_FILE = "../output"
PATH_VALID_FILE = "../output"
PATH_TEST_FILE = "../output"
X_train=np.load(os.path.join(PATH_TRAIN_FILE, 'X_train.npy'))
Y_train=np.load(os.path.join(PATH_TRAIN_FILE, 'Y_train.npy'))
X_valid=np.load(os.path.join(PATH_VALID_FILE, 'X_valid.npy'))
Y_valid=np.load(os.path.join(PATH_VALID_FILE, 'Y_valid.npy'))
X_test=np.load(os.path.join(PATH_TEST_FILE, 'X_test.npy'))
Y_test=np.load(os.path.join(PATH_TEST_FILE, 'Y_test.npy'))


torch.manual_seed(0)
if torch.cuda.is_available():
	torch.cuda.manual_seed(0)
    
    
MODEL_TYPE = 'RFC'  # TODO: Change this to 'MLP', 'CNN', or 'RNN' according to your task
s='CNN' 'MLP' 'RNN''LSTM'
if MODEL_TYPE in s:
    NUM_EPOCHS = 15
    BATCH_SIZE = 100
    USE_CUDA = False  # Set 'True' if you want to use GPU
    NUM_WORKERS = 0
    train_dataset = load_dataset(X_train,Y_train, MODEL_TYPE)
    valid_dataset = load_dataset(X_valid,Y_valid, MODEL_TYPE)
    test_dataset = load_dataset(X_test,Y_test, MODEL_TYPE)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


 
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
     
    np.random.seed(1000)
    weights = [1, 2, 1, 1, 1.1]
    class_weights = torch.FloatTensor(weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
    print("\nAccuracy and F1 score on test data: %0.2f %0.2f with %s "  % (acc_model,f1score_model,MODEL_TYPE))
    
elif MODEL_TYPE == 'RFC':
    '''best parameter from grid search'''
    rfc=RandomForestClassifier(n_estimators=600, random_state=42,min_samples_split=5,min_samples_leaf=1,max_features='auto',max_depth=80)
    rfc.fit(X_train, Y_train)
    Y_pred=rfc.predict(X_train)
    YP_test=rfc.predict(X_test)
    acc=accuracy_score(YP_test,Y_test)
    acc
    f1score=f1_score(YP_test,Y_test,average="weighted")
    class_names = ['Wake', 'Stage 1', 'Stage 2', 'Stage 3', 'R']
    plot_confusion_matrix(list(map(lambda x,y:(x,y),Y_test,YP_test)), class_names)
    print("\nAccuracy and F1 score on test data: %0.2f %0.2f with %s "  % (acc,f1score,MODEL_TYPE))
    # save the model to disk
    save_file = 'RFC.sav'
    pickle.dump(rfc, open(os.path.join(PATH_OUTPUT, save_file), 'wb'))
    
else:
	raise AssertionError("Wrong Model Type!")   
    



