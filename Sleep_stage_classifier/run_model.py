# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:36:31 2019

@author: CRNZ
"""
import os
import sys
import numpy as np
from plots import plot_learning_curves, plot_confusion_matrix
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer    
from sklearn.preprocessing import StandardScaler
import pickle

PATH_OUTPUT = "./model/"
os.makedirs(PATH_OUTPUT, exist_ok=True)


# Set a correct path to the seizure data file you downloaded
PATH_TRAIN_FILE = "./output"
PATH_VALID_FILE = "./output"
PATH_TEST_FILE = "./output"
X_train=np.load(os.path.join(PATH_TRAIN_FILE, 'X_train.npy'))
Y_train=np.load(os.path.join(PATH_TRAIN_FILE, 'Y_train.npy'))
X_valid=np.load(os.path.join(PATH_VALID_FILE, 'X_valid.npy'))
Y_valid=np.load(os.path.join(PATH_VALID_FILE, 'Y_valid.npy'))
X_test=np.load(os.path.join(PATH_TEST_FILE, 'X_test.npy'))
Y_test=np.load(os.path.join(PATH_TEST_FILE, 'Y_test.npy'))


'''best parameter from grid search'''
rfc=RandomForestClassifier(n_estimators=600, random_state=42,min_samples_split=5,min_samples_leaf=1,max_features='auto',max_depth=80)
rfc.fit(X_train, Y_train)
Y_pred=rfc.predict(X_train)
YP_test=rfc.predict(X_test)
acc=accuracy_score(YP_test,Y_test)
f1score=f1_score(YP_test,Y_test,average="weighted")
class_names = ['Wake', 'Stage 1', 'Stage 2', 'Stage 3', 'R']
plot_confusion_matrix(list(map(lambda x,y:(x,y),Y_test,YP_test)), class_names)
print("\nAccuracy and F1 score on test data: %0.2f %0.2f"  % (acc,f1score))
# save the model to disk
save_file = 'RFC.sav'
pickle.dump(rfc, open(os.path.join(PATH_OUTPUT, save_file), 'wb'))
    
 
    



