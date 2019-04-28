from sklearn.cluster import spectral_clustering
import pandas as pd
import numpy as np


def create_model():
    data = pd.read_csv('output/SubjectSimilarities.scala.csv', delimiter=',', encoding="utf-8-sig")
    print(data.head(5))
    print(data.columns)
    n = max(data['_1'].unique()) + 1
    matrix = np.zeros([n, n])

    for index, row in data.iterrows():
        matrix[int(row['_1']), int(row['_2'])-1] = 1 - float(row['_3'])
        matrix[int(row['_2'])-1, int(row['_1'])] = 1 - float(row['_3'])
        if index % 10000 == 0:
            print("{0:.0%}".format(index / data.shape[0]))

    print("starting clustering...")
    labels = spectral_clustering(matrix, n_clusters=5, eigen_solver='arpack')

    return labels


def write_output(output):
    import csv

    print("Writing output...")

    with open('output/ClusterID.csv', mode='w') as employee_file:
        W = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        W.writerow(['patientID', 'clusterID'])
        for i in range(len(output)):
            W.writerow([i, output[i]])



if __name__ == "__main__":
    print("Executing as Main")
    assignments = create_model()
    print(assignments)
    write_output(assignments)


