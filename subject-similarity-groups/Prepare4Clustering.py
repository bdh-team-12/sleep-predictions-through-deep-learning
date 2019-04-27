import pandas as pd
import numpy as np


def get_input():
    subjects = pd.read_csv('data/SUBJECTS.csv')
    demographics = pd.read_csv('data/DEMOGRAPHICS.csv')
    medical_history = pd.read_csv('data/MEDICAL_HISTORY.csv')
    medication = pd.read_csv('data/MEDICATION.csv')

    demographics['value'] = demographics['value'].astype('int32')

    dem_vars = demographics['classifier'].unique()
    ages = list(np.sort(demographics[demographics['classifier'] == "age_category_s1"]['value'].unique()))
    educat = list(np.sort(demographics[demographics['classifier'] == "educat"]['value'].unique()))
    ethn = list(np.sort(demographics[demographics['classifier'] == "ethnicity"]['value'].unique()))
    gender = list(np.sort(demographics[demographics['classifier'] == "gender"]['value'].unique()))
    mstat = list(np.sort(demographics[demographics['classifier'] == "mstat"]['value'].unique()))
    race = list(np.sort(demographics[demographics['classifier'] == "race"]['value'].unique()))
    min_age = min(ages); min_ed = min(educat); min_eth = min(ethn); min_gen = min(gender); min_st = min(mstat)
    min_r = min(race)
    meh_vars = list(medical_history['condition'].unique())
    med_vars = list(medication['medicine'].unique())
    nsrrid = list(np.sort(subjects['nsrrid'].unique()))

    print("Input has been read into memory")

    K = [len(ages)]
    K.append(K[0] + len(educat))
    K.append(K[1] + len(ethn))
    K.append(K[2] + len(gender))
    K.append(K[3] + len(mstat))
    K.append(K[4] + len(race))

    dem_graph = np.zeros([len(nsrrid), max(K)], dtype=bool)
    for index, row in demographics.iterrows():
        val = int(row['value'])
        if val != 0:
            p = nsrrid.index(row['nsrrid'])
            col = row['classifier']
            if col == "age_category_s1":
                dem_graph[p, val - min_age] = 1
            if col == "educat":
                dem_graph[p, K[0] + val - min_ed] = 1
            if col == "ethnicity":
                dem_graph[p, K[1] + val - min_eth] = 1
            if col == "gender":
                dem_graph[p, K[2] + val - min_gen] = 1
            if col == "mstat":
                dem_graph[p, K[3] + val - min_st] = 1
            if col == "race":
                dem_graph[p, K[4] + val - min_r] = 1

        if index % 9000 == 0:
            print("Reformatting Demographics...")

    temp = medication[medication['value'] != 0]

    med_graph = np.zeros([len(nsrrid), len(med_vars)], dtype=bool)
    for index, row in temp.iterrows():
        p = nsrrid.index(row['nsrrid'])
        col = med_vars.index(row['medicine'])
        med_graph[p, col] = 1
        if index % 9000 == 0:
            print("Reformatting Medications...")

    tempH = medical_history[medical_history['value'] != 0]
    print(tempH.shape)

    meh_graph = np.zeros([len(nsrrid), len(meh_vars)], dtype=bool)
    for index, row in tempH.iterrows():
        p = nsrrid.index(row['nsrrid'])
        col = meh_vars.index(row['condition'])
        meh_graph[p, col] = 1
        if index % 9000 == 0:
            print("Reformatting Medical History...")

    temp = np.hstack((dem_graph, med_graph))
    graph = np.hstack((temp, meh_graph))

    return nsrrid, graph


def do_stuff(ids, graph):
    integer = graph.astype(int)

    intersection = np.matmul(integer, np.transpose(integer))
    union = np.zeros(intersection.shape)
    k = 0

    for row in graph:
        union[k, :] = np.sum(row + graph, axis=1)
        k = k + 1

    np.seterr(divide='ignore', invalid='ignore')
    similarities = np.divide(intersection, union)

    print("Calculated Similarity Matrix")

    n = intersection.shape[0]
    nrow = (n*(n+1))/2
    result = np.zeros([int(nrow), 3])
    k = 0
    for i in range(intersection.shape[0]):
        for j in range(i):
            result[k, :] = [ids[i], ids[j], similarities[i, j]]
            k = k + 1

    return result


def write_output(output):
    import csv

    print("Writing output...")

    with open('output/SubjectSimilarities.scala.csv', mode='w') as employee_file:
        W = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        W.writerow(['subject_id1', 'subject_id2', 'similarity'])
        for i in range(output.shape[0]):
            W.writerow([output[i, 0].astype(int), output[i, 1].astype(int), output[i, 2]])


if __name__ == "__main__":
    print('Executing as Main')
    people, answer = get_input()
    print("Input has been processed successfully")
    out = do_stuff(people, answer)
    print("Similarities have been computed successfully")
    write_output(out)
    print("File has been written successfully")
