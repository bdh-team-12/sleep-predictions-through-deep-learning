# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:16:19 2019

@author: CRNZ
"""
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

'''load presaved dataset'''
basedir='C:/Users/CRNZ/Documents/GitHub/sleep-predictions-through-deep-learning/mean PSD'

X_train=np.load(os.path.join(basedir, 'X_train.npy'))
Y_train=np.load(os.path.join(basedir, 'Y_train.npy'))
X_valid=np.load(os.path.join(basedir, 'X_valid.npy'))
Y_valid=np.load(os.path.join(basedir, 'Y_valid.npy'))
X_test=np.load(os.path.join(basedir, 'X_test.npy'))
Y_test=np.load(os.path.join(basedir, 'Y_test.npy'))

num_classes=len(set(Y_train))

Y_train2 = np_utils.to_categorical(Y_train, num_classes)

model_m = Sequential()
model_m.add(Reshape((36, 1), input_shape=(36,)))
model_m.add(Conv1D(100, 2, activation='relu',input_shape=(36,1)))
model_m.add(Conv1D(100, 2, activation='relu'))
model_m.add(MaxPooling1D(2))
model_m.add(Conv1D(160, 2, activation='relu'))
model_m.add(Conv1D(160, 2, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())


callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# Initial model

BATCH_SIZE = 200
EPOCHS = 50

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model_m.fit(X_train,
                      Y_train2,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.1,
                      verbose=0)

# summarize history for accuracy and loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['acc'], "g--", label="Accuracy of training data")
plt.plot(history.history['val_acc'], "g", label="Accuracy of validation data")
plt.plot(history.history['loss'], "r--", label="Loss of training data")
plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

#%%

print("\n--- Check against test data ---\n")


# Set input_shape / reshape for Keras

X_test2 = X_test.astype("float32")
Y_test2 = Y_test.astype("float32")
Y_test2 = np_utils.to_categorical(Y_test2, num_classes)
score = model_m.evaluate(X_test2, Y_test2, verbose=1)
print("\nAccuracy on test data: %0.2f" % score[1])
print("\nLoss on test data: %0.2f" % score[0])
# Hyperparameter

'''grid search - best epochs 100, batch size 200'''
batch_size = [100, 200, 300, 400, 500]
epochs = [10, 50, 100]

for BATCH_SIZE in batch_size:
    for EPOCHS in epochs:
        history = model_m.fit(X_train,
                      Y_train2,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.1,
                      verbose=0)
        X_test2 = X_test.astype("float32")
        Y_test2 = Y_test.astype("float32")
        Y_test2 = np_utils.to_categorical(Y_test2, num_classes)
        score = model_m.evaluate(X_test2, Y_test2, verbose=1)

        print("\nAccuracy on test data: %0.2f with 'epochs':%r,'batch_size':%r" % (score[1],EPOCHS,BATCH_SIZE))
        print("\nLoss on test data: %0.2f" % score[0])

'''grid search - best epochs 100, batch size 200'''        
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BATCH_SIZE=200
epochs=100
for dropout in dropout_rate:
    model_m = Sequential()
    model_m.add(Reshape((36, 1), input_shape=(36,)))
    model_m.add(Conv1D(100, 2, activation='relu',input_shape=(36,1)))
    model_m.add(Conv1D(100, 2, activation='relu'))
    model_m.add(MaxPooling1D(2))
    model_m.add(Conv1D(160, 2, activation='relu'))
    model_m.add(Conv1D(160, 2, activation='relu'))
    model_m.add(GlobalAveragePooling1D())
    model_m.add(Dropout(dropout))
    model_m.add(Dense(num_classes, activation='softmax'))
    print(model_m.summary())


    callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
    ]

    model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
    history = model_m.fit(X_train,
                      Y_train2,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.1,
                      verbose=0)
    X_test2 = X_test.astype("float32")
    Y_test2 = Y_test.astype("float32")
    Y_test2 = np_utils.to_categorical(Y_test2, num_classes)
    score = model_m.evaluate(X_test2, Y_test2, verbose=1)    
    print("\nAccuracy and loss on test data: %0.2f %0.2f with 'dropout':%r" % (score[1],score[0], dropout))


'''
def create_model(dropout_rate=0.0):
	# create model
	model_m = Sequential()
	model_m.add(Reshape((36, 1), input_shape=(36,)))
	model_m.add(Conv1D(100, 2, activation='relu',input_shape=(36,1)))
	model_m.add(Conv1D(100, 2, activation='relu'))
	model_m.add(MaxPooling1D(2))
	model_m.add(Conv1D(160, 2, activation='relu'))
	model_m.add(Conv1D(160, 2, activation='relu'))
	model_m.add(GlobalAveragePooling1D())
	model_m.add(Dropout(dropout_rate))
	model_m.add(Dense(num_classes, activation='softmax'))
	callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
	]

	model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
	return model_m
    
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=200, verbose=1,validation_split=0.1)
# define the grid search parameters
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(dropout_rate=dropout_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, Y_train2)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
'''