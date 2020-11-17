import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biosppy.signals.ecg as ecg


# Read the data from csv file
X_train = pd.read_csv(r"X_train.csv")
X_test  = pd.read_csv(r"X_test.csv")
# Drop the id columns
X_train = X_train.drop(columns= 'id', axis = 1)
X_test = X_test.drop(columns= 'id', axis = 1)
# Get number of samples
numberOfTrainSamples = X_train.shape[0]
numberOfTestSamples = X_test.shape[0]

print("Number of train samples:", numberOfTrainSamples)
print("Number of test samples: ", numberOfTestSamples)

all_templates = np.empty([ 0,180 ])

#all_templates = []

print("Starting with train data")
for i in range(numberOfTrainSamples):
   
    currentPatient = X_train.iloc[i].dropna().values
    
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(currentPatient, sampling_rate=300, show=False)
    templates, rpeaks_ = ecg.extract_heartbeats(filtered, rpeaks = rpeaks, sampling_rate=300)

    all_templates = np.append(all_templates, templates, axis=0)
    if i % 500 == 0:
        print("i is ", i)
        print('templates shape: ', templates.shape)
        print("all_templates shape: ", all_templates.shape)
    #all_templates = np.vstack((all_templates, templates))
    #all_templates.append(templates)

print("Starting with test data")
for i in range(numberOfTestSamples):

    currentPatient = X_test.iloc[i].dropna().values

    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(currentPatient, sampling_rate=300, show=False)
    templates, rpeaks_ = ecg.extract_heartbeats(filtered, rpeaks = rpeaks, sampling_rate=300)

    all_templates = np.append(all_templates, templates, axis=0)
    #all_templates.append(templates)
    if i % 500 == 0:
        print("i is ", i)
        print('templates shape: ', templates.shape)
        print("all_templates shape: ", all_templates.shape)


print()
#all_templates = np.asarray(all_templates)
print("Final shape of all_templates is: ", all_templates.shape)
np.save('all_heartbeats_eh_filtered', all_templates)

# the file called all_heartbeats contains the templates which are returned by ecg
# the file called all_heartbeats_eh contains the tmeplates which are return by extract_heartbeats

# they are in fact different and no idea which one is correct... maybe even none of them..