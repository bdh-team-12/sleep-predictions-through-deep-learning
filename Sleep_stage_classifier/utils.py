import os
import time
import numpy as np
import pandas as pd
import torch
import mne
from mne.time_frequency import psd_array_welch
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
    dffinal=pd.concat([df,df2,df3], axis=1)
    return dffinal.values


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def compute_batch_accuracy(output, target):
	"""Computes the accuracy for a batch"""
	with torch.no_grad():

		batch_size = target.size(0)
		_, pred = output.max(1)
		correct = pred.eq(target).sum()

		return correct * 100.0 / batch_size


def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	model.train()

	end = time.time()
	for i, (input, target) in enumerate(data_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)
		target = target.to(device)

		optimizer.zero_grad()
		output = model(input)
		loss = criterion(output, target)
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		losses.update(loss.item(), target.size(0))
		accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

		if i % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
				epoch, i, len(data_loader), batch_time=batch_time,
				data_time=data_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg


def evaluate(model, device, data_loader, criterion, print_freq=10):
	batch_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	results = []

	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(data_loader):

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)

			output = model(input)
			loss = criterion(output, target)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			losses.update(loss.item(), target.size(0))
			accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

			y_true = target.detach().to('cpu').numpy().tolist()
			y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
			results.extend(list(zip(y_true, y_pred)))

			if i % print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					i, len(data_loader), batch_time=batch_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg, results


def make_kaggle_submission(list_id, list_prob, path):
	if len(list_id) != len(list_prob):
		raise AttributeError("ID list and Probability list have different lengths")

	os.makedirs(path, exist_ok=True)
	output_file = open(os.path.join(path, 'my_predictions.csv'), 'w')
	output_file.write("SUBJECT_ID,MORTALITY\n")
	for pid, prob in zip(list_id, list_prob):
		output_file.write("{},{}\n".format(pid, prob))
	output_file.close()
