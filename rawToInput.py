import numpy as np
import pandas as pd
from numpy import genfromtxt
import math
np.set_printoptions(suppress=True)


dictionary = pd.read_csv('shhs-data-dictionary-0.13.2-variables.csv')
dems = dictionary['folder'] == "Demographics"
Demgraphics = [item.lower() for item in dictionary[dems]['id'].values]
med = dictionary['folder'] == "Medical History"
medHist = [item.lower() for item in dictionary[med]['id'].values]
meds = dictionary['folder'] == "Medications"
medics = [item.lower() for item in dictionary[meds]['id'].values]
print(medics)


mortality_data = genfromtxt('shhs-cvd-summary-dataset-0.13.0.csv', delimiter=',')
data = genfromtxt('shhs1-dataset-0.13.0.csv', delimiter=',', skip_header=1)

import csv
with open('shhs1-dataset-0.13.0.csv', newline='') as f:
    reader = csv.reader(f)
    names = next(reader)

names = [item.lower() for item in names]
print(names)


DEMOGRAPHICS = [ names.index(elem) if elem in names else -1 for elem in Demgraphics ]
print(DEMOGRAPHICS)
indices = [i for (i, v) in enumerate(DEMOGRAPHICS) if v < 2]
DEMOGRAPHICS = np.array(np.delete(DEMOGRAPHICS, indices, 0), dtype='int')
print(DEMOGRAPHICS)


MED_HISTORY = [ names.index(elem) if elem in names else -1 for elem in medHist ]
indices = [i for (i, v) in enumerate(MED_HISTORY) if v < 2]
MED_HISTORY = np.delete(MED_HISTORY, indices, 0)
MEDICATIONS = [ names.index(elem) if elem in names else -1 for elem in medics ]
indices = [i for (i, v) in enumerate(MEDICATIONS) if v < 2]
MEDICATIONS = np.delete(MEDICATIONS, indices, 0)

names = np.array(names)
print(names[MEDICATIONS])
my_list = list()
my_list.append(1)
my_list.extend(DEMOGRAPHICS)
print(np.array(my_list))
DEM_OUT = data[:, np.array(my_list)]
print(DEM_OUT)


HIST_OUT = np.zeros([int(1e7), 2], dtype='int')
MEDS_OUT = np.zeros([int(1e7), 2], dtype='int')
row1 = 0
row2 = 0
row3 = 0
print(data[:5, :5])
for i in range(data.shape[0]):
    for k, j in enumerate(MED_HISTORY):
        if not np.isnan(data[i, j]):
            if data[i, j] > 0:
                HIST_OUT[row2, 0] = data[i, 1]
                HIST_OUT[row2, 1] = k + 1
                row2 = row2 + 1

    for k, j in enumerate(MEDICATIONS):
        if not np.isnan(data[i, j]):
            if data[i, j] > 0:
                MEDS_OUT[row3, 0] = data[i, 1]
                MEDS_OUT[row3, 1] = k + 1
                row3 = row3 + 1

print("___")
print(row3)
DEM_OUT = np.array(DEM_OUT, dtype='int')
print(DEM_OUT[:15, :])

print(names[my_list])

row2 = row2 - 1
row3 = row3 - 1

HIST_OUT = HIST_OUT[:row2, :]
MEDS_OUT = MEDS_OUT[:row3, :]

import csv

with open('DEMOGRAPHICS.csv', mode='w') as employee_file:
    W = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    W.writerow(['patientID', 'age_category', 'age', 'education', 'ethnicity', 'gender', 'marriage', 'race'])
    for i in range(DEM_OUT.shape[0]):
        W.writerow(DEM_OUT[i, :])

with open('MEDICAL_HISTORY.csv', mode='w') as employee_file:
    W = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    W.writerow(['patientID', 'eventID'])
    for i in range(HIST_OUT.shape[0]):
        W.writerow(HIST_OUT[i, :])

with open('MEDICATION.csv', mode='w') as employee_file:
    W = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    W.writerow(['patientID', 'medicationID'])
    for i in range(MEDS_OUT.shape[0]):
        W.writerow(MEDS_OUT[i, :])
