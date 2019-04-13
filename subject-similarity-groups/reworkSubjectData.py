import os
import pickle
import pandas as pd
import sys

df_raw = pd.read_csv('shhs1-dataset-0.13.0.csv')
df_demographic = df_raw[['age_category_s1', 'age_s1', 'educat', 'ethnicity', 'gender', 'mstat', 'nsrrid', 'pptid', 'race']].dropna()
df_medical_history = df_raw[['prev_hx_stroke','angina15','ca15','cabg15','copd15','hf15','mi15','othrcs15','pacem15','prev_hx_mi','parrptdiab']].dropna()
df_medication = df_raw[['ace1', 'aced1', 'anar1a1', 'anar1b1', 'anar1c1', 'anar31', 'benzod1', 'beta1', 'betad1', 'ccb1', 'diuret1', 'htnmed1', 'insuln1', 'istrd1', 'lipid1', 'nsaid1', 'ntca1', 'ntg1', 'ohga1', 'ostrd1', 'tca1']].dropna()
# df_measurment = df_raw[['bmi_s1', 'height', 'hip', 'waist', 'weight', 'diasbp', 'systbp', 'chol', 'hdl', 'trig', 'fev1', 'fvc', 'ai_all', 'ahi_a0h3', 'ahi_a0h3a', 'ahi_a0h4', 'ahi_a0h4a', 'ahi_c0h3', 'ahi_c0h3a', 'ahi_c0h4', 'ahi_c0h4a', 'ahi_o0h3', 'ahi_o0h3a', 'ahi_o0h4', 'ahi_o0h4a', 'cai0p', 'oahi', 'oai0p', 'rdi0p', 'rdi3p', 'rdi3pa', 'rdi4p', 'rdinr3p', 'rdirem3p', 'nsupinep', 'pctlt75', 'pctlt80', 'pctlt85', 'pctlt90', 'slp_eff', 'slp_lat', 'slpprdp', 'slptime', 'supinep', 'times34p', 'timest1p', 'timest2p', 'waso']].dropna()

print(df_demographic)

def write_to_csv(row, fileName):
    with open(fileName + '.tmp', 'a') as data:
        for i in range(0, len(row.index)):
            data.write(str(row.index[i]) + ',' + row.name + ',' + str(row.values[i]) + '\n')
        print(row.name)
    # remove duplicate lines
    lines_seen = set() # holds lines already seen
    outfile = open(fileName + '.csv', "w")
    for line in open(fileName + '.tmp', "r"):
        if line not in lines_seen: # not a duplicate
            outfile.write(line)
            lines_seen.add(line)
    outfile.close()

def write_df_demographic_to_csv(row):
    write_to_csv(row, "DEMOGRAPHICS")

def write_df_medical_history_to_csv(row):
    write_to_csv(row, "MEDICAL_HISTORY")

def write_df_medication_to_csv(row):
    write_to_csv(row, "MEDICATION")

# def write_df_measurment_to_csv(row):
#     write_to_csv(row, "MEASURMENTS")

    # for i in row:
    #     print(i)

#df_measurment.applymap(write_to_csv)

t = df_demographic.apply(write_df_demographic_to_csv)
t = df_medical_history.apply(write_df_medical_history_to_csv)
t = df_medication.apply(write_df_medication_to_csv)
# t = df_measurment.apply(write_df_measurment_to_csv)

#print(t)
