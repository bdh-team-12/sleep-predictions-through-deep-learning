import pandas as pd
import numpy as np
import csv
np.set_printoptions(precision=3)

# Given a

def do_stuff(fil):
    inputt = pd.read_csv('output/ClusterID.csv')
    clusters = np.sort(inputt['clusterID'].unique())  # we need to know the cluster numbers.

    with open('data/shhs-cvd-summary-dataset-0.13.0.csv', newline='') as f:
        reader = csv.reader(f)
        names = next(reader)

    cols = [names.index(elem) if elem in names else -1 for elem in fil]
    indices = [i for (i, v) in enumerate(cols) if v < 0]
    cols = np.array(np.delete(cols, indices, 0), dtype='int')
    c = len(cols) + 1
    death_cols = [12, 13, 23, 9, 11]

    data = np.genfromtxt('data/shhs-cvd-summary-dataset-0.13.0.csv', delimiter=',', skip_header=1)
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

def process_file(filename):
    # Overwrite the file, removing empty lines
    with open(filename, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.writelines(line for line in lines if line.strip())
        f.truncate()
    with open(filename, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.writelines(lines[0])
        for line in lines[1:]:
            f.writelines(str(int(float(line.split(",")[0]))) + "," + ",".join(line.split(",")[1:]))
        f.truncate()

def write_output(stuff, h):
    print("Writing output...")

    with open('output/ClusterOutcomes.csv', mode='w') as cluster_file:
        W = csv.writer(cluster_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        W.writerow(h)
        for i in range(0, stuff.shape[0]):
            W.writerow(stuff[i, :])
    process_file('output/ClusterOutcomes.csv')
    print("Success")

def cluster_risk_factors():
    cluster_outcomes = pd.read_csv('output/ClusterOutcomes.csv')
    rows = []
    with open('output/ClusterSimilarities.csv', mode='r') as similarities:
        lines = similarities.readlines()
        for line in lines[1:]:
            cluster_id = int(line.split(",")[0])
            match = float(line.split(",")[1])
            row = cluster_outcomes.loc[cluster_outcomes['ClusterID'] == cluster_id]
            row = row.apply(lambda r: r * match * 100)
            rows.append(row)
            # TODO make this work to find the sum of each CVD risk factor
            # cvd_results[cluster_id] = (match * row["Coronary Heart Disease"].values[0] * 100,
            #                            match * row["Congestive Heart Failure"].values[0] * 100,
            #                            match * row["Coronary Artery Bypass Graft Surgeries"].values[0] * 100,
            #                            match * row["Congestive Heart Failure"].values[0] * 100,
            #                            match * row["Myocardial Infractions"].values[0] * 100,
            #                            match * row["Stroke"].values[0] * 100,
            #                            match * row["is_alive"].values[0] * 100)
    for i in range(len(rows)):
        print(i)

if __name__ == "__main__":
    csd_types = ['any_chd', 'any_cvd', 'cabg', 'chf', 'mi', 'stroke', 'vital']
    data = do_stuff(csd_types)
    header = ['ClusterID', 'Coronary Heart Disease', 'Congestive Heart Failure',
              'Coronary Artery Bypass Graft Surgeries', 'Congestive Heart Failure', 'Myocardial Infractions', 'Stroke',
              'is_alive']
    write_output(data, header)
    print("Finding risk factors...")
    cluster_risk_factors()
