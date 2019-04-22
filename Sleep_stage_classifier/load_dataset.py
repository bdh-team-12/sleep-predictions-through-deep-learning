# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:17:27 2019

@author: CRNZ
"""

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset

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