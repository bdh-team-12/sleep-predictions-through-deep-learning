import pandas as pd
import numpy as np
import csv
np.set_printoptions(precision=3)


def do_stuff(fil):
    inputt = pd.read_csv('ClusterID.csv')
    clusters = np.sort(inputt['clusterID'].unique())  # we need to know the cluster numbers.

    with open('shhs-cvd-summary-dataset-0.13.0.csv', newline='') as f:
        reader = csv.reader(f)
        names = next(reader)

    cols = [names.index(elem) if elem in names else -1 for elem in fil]
    indices = [i for (i, v) in enumerate(cols) if v < 0]
    cols = np.array(np.delete(cols, indices, 0), dtype='int')
    c = len(cols) + 1
    death_cols = [12, 13, 23, 9, 11]

    data = np.genfromtxt('shhs-cvd-summary-dataset-0.13.0.csv', delimiter=',', skip_header=1)
    output = np.zeros([len(clusters), c])

    i = 0
    for k in clusters:
        idx = inputt['clusterID'] == k
        p = inputt[idx]['patientID'].values
        idc = np.in1d(data[:, 1], p)
        temp = data[idc, :]
        temp = np.nan_to_num(temp[:, cols])
        temp = np.minimum(temp, 1)
        output[i, 1:c] = np.mean(temp, axis=0)
        output[i, 0] = k
        i = i + 1

    return output


def write_output(stuff, h):
    print("Writing output...")

    with open('DEMOGRAPHICS.csv', mode='w') as employee_file:
        W = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        W.writerow(h)
        for i in range(stuff.shape[0]):
            W.writerow(stuff[i, :])


if __name__ == "__main__":
    use = ['angina', 'any_chd', 'any_cvd', 'cabg', 'chf', 'mi', 'stroke', 'vital']
    data = do_stuff(use)
    header = ['ClusterID', 'Chest Pain', 'Coronary Heart Disease', 'Congestive Heart Failure',
              'Coronary Artery Bypass Graft Surgeries', 'Congestive Heart Failure', 'Myocardial Infractions', 'Stroke',
              'is_alive']
    write_output(data, header)
