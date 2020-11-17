import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biosppy.signals.ecg as ecg

from sklearn.preprocessing import Normalizer

import keras
from keras.models import load_model


extract_heartbeat = True

if extract_heartbeat:
    train_filename = 'all_heartbeats_eh_filtered.npy'
    encoder_filename = 'encoder_eh.h5'
    data_filename = 'preprocessed_data_eh'
else:
    train_filename = 'all_heartbeats.npy'
    encoder_filename = 'encoder.h5'
    data_filename = 'preprocessed_data'






encoder = load_model(encoder_filename)
normalizer = Normalizer()
# Read the data from csv file
X_train = pd.read_csv(r"X_train.csv")
X_all = np.load(train_filename)
normalizer.fit(X_all)
X_test  = pd.read_csv(r"X_test.csv")
# Drop the id columns
X_train = X_train.drop(columns= 'id', axis = 1)
X_test = X_test.drop(columns= 'id', axis = 1)
# Get number of samples
#X_train = normalizer.transform(X_train)


numberOfTrainSamples = X_train.shape[0]
numberOfTestSamples = X_test.shape[0]

print('Starting with train data')
all_aggregations = np.empty([numberOfTrainSamples + numberOfTestSamples,120])

for i in range(numberOfTrainSamples):

    currentPatient = X_train.iloc[i].dropna().values
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(currentPatient, sampling_rate=300, show=False)
    templates, rpeaks_ = ecg.extract_heartbeats(filtered, rpeaks = rpeaks, sampling_rate=300)

    number_templates = templates.shape[0]
    
    templates = normalizer.transform(templates)
    encoded_templates = encoder.predict(templates)
    averaged_encoded_templates = np.sum(encoded_templates, axis=0) / number_templates
    

    averaged_encoded_templates = averaged_encoded_templates.reshape(1, -1)
    if i % 500 == 0:
        print("i is ", i)
        print('number of templates: ', number_templates)
        print("Shape encoded templates before sum out: ", encoded_templates.shape)
        print('shape of averaged_encoded_templates after sum out: ', averaged_encoded_templates.shape)
    #all_aggregations = np.append(all_aggregations, averaged_encoded_templates, axis=0)
    all_aggregations[i] = averaged_encoded_templates
    if i % 500 == 0:
        print('shape of all_aggregations after sum out: ', all_aggregations.shape)

print('shape of final train data is ', all_aggregations.shape)



for i in range(numberOfTestSamples):
    
    currentPatient = X_test.iloc[i].dropna().values
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(currentPatient, sampling_rate=300, show=False)
    templates, rpeaks_ = ecg.extract_heartbeats(filtered, rpeaks = rpeaks, sampling_rate=300)

    number_templates = templates.shape[0]
    
    
    templates = normalizer.transform(templates)
    encoded_templates = encoder.predict(templates)
    averaged_encoded_templates = np.sum(encoded_templates, axis=0) / number_templates
    

    averaged_encoded_templates = averaged_encoded_templates.reshape(1, -1)
    if i % 500 == 0:
        print("i is ", i)
        print('number of templates: ', number_templates)
        print("Shape encoded templates before sum out: ", encoded_templates.shape)
        print('shape of averaged_encoded_templates after sum out: ', averaged_encoded_templates.shape)
    #all_aggregations = np.append(all_aggregations, averaged_encoded_templates, axis=0)
    all_aggregations[i + numberOfTrainSamples] = averaged_encoded_templates
    
print('shape of final train data is ', all_aggregations.shape)


np.save(data_filename, all_aggregations)

 

















    #averaged_encoded_templates = np.empty([0, 60])
    # for j in range(number_templates):

    #     currentTemplate = templates2[j,:]
    #     currentTemplate = currentTemplate.reshape(1, -1)
    #     currentTemplate = normalizer.transform(currentTemplate)    
    #     #print('safas ', currentTemplate.shape)    
    #     encodedTemplate = encoder.predict(currentTemplate)
    #     averaged_encoded_templates = np.append(averaged_encoded_templates, encodedTemplate, axis=0)
    #     if i % 100 == 0 and j == number_templates-1:
    #         print('shape of curr temp: ', currentTemplate.shape)
    #         print('shape of encoded temp: ', encodedTemplate.shape)
    #         print('shape of averaged encoded templates: ', averaged_encoded_templates.shape)
