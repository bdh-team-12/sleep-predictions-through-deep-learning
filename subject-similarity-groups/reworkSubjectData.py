import pandas as pd
import os


# This script converts the shhs1-dataset into separate csv files based on variable categories used for sparkx graphing

df_raw = pd.read_csv('data/shhs1-dataset-0.13.0.csv')
df_raw.columns = map(str.lower, df_raw.columns)
df_subject = df_raw[['pptid']]
df_demographic = df_raw[['age_category_s1', 'age_s1', 'educat', 'ethnicity', 'gender', 'mstat', 'nsrrid', 'pptid', 'race']].dropna()
df_medical_history = df_raw[['prev_hx_stroke','angina15','ca15','cabg15','copd15','hf15','mi15','othrcs15','pacem15','prev_hx_mi','parrptdiab']].dropna()
df_medication = df_raw[['ace1', 'aced1', 'anar1a1', 'anar1b1', 'anar1c1', 'anar31', 'benzod1', 'beta1', 'betad1', 'ccb1', 'diuret1', 'htnmed1', 'insuln1', 'istrd1', 'lipid1', 'nsaid1', 'ntca1', 'ntg1', 'ohga1', 'ostrd1', 'tca1']].dropna()

def write_to_csv(row, fileName,header):
    with open(fileName + '.tmp', 'a') as data:
        for i in range(0, len(row.index)):
            data.write(str(row.index[i]) + ',' + row.name + ',' + str(row.values[i]) + '\n')
        print(row.name)
    # remove duplicate lines
    lines_seen = set() # holds lines already seen
    outfile = open(fileName + '.csv', "w")
    outfile.write(header)
    for line in open(fileName + '.tmp', "r"):
        if line not in lines_seen: # not a duplicate
            outfile.write(line)
            lines_seen.add(line)
    os.remove(fileName + '.tmp')
    outfile.close()

def write_df_demographic_to_csv(row):
    write_to_csv(row, "data/DEMOGRAPHICS",'nsrrid,classifier,value\n')

def write_df_medical_history_to_csv(row):
    write_to_csv(row, "data/MEDICAL_HISTORY",'nsrrid,condition,value\n')

def write_df_medication_to_csv(row):
    write_to_csv(row, "data/MEDICATION",'nsrrid,medicine,value\n')
    
def write_df_subject_to_csv(row):
    write_to_csv(row, "data/SUBJECTS",'nsrrid,idtype,pptid\n')
    
if __name__ == "__main__":
    print('Executing as Main')
    t = df_demographic.apply(write_df_demographic_to_csv)
    print("df_demographic has been written successfully")
    t = df_medical_history.apply(write_df_medical_history_to_csv)
    print("df_medical_history has been written successfully")
    t = df_medication.apply(write_df_medication_to_csv)
    print("df_medication has been written successfully")
    t = df_subject.apply(write_df_subject_to_csv)
    print("df_subject has been written successfully")
