import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biosppy.signals.ecg as ecg

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import neurokit as nk
import auxilary
import nkcopy
import frequency_analysis

print('Reading data')
X_train = pd.read_csv(r'X_test.csv')
print('done with reading')
del X_train['id']

heartbeat1 = np.load('one_hb.npy')
preprocessed_heartbeat1 = nk.ecg_preprocess(ecg=heartbeat1, sampling_rate=300)

numberOfTrainingSamples = 5117
numberOfTestSamples = X_train.values.shape[0]
print('number of test samples: ', numberOfTestSamples)
numberOfFeatures = 43

preprocessed_X_train = np.empty([numberOfTestSamples, numberOfFeatures])

features_names_timehvr = ['sdNN', 'meanNN', 'CVSD', 'cvNN', 'RMSSD', 'medianNN',
                            'madNN', 'mcvNN', 'pNN50', 'pNN20']

# features_names_freqhvr = ['Triang', 'Shannon_h', 'VLF', 'LF', 'HF', 'Total_Power',
#                             'LF/HF', 'LFn', 'HFn', 'LF/P', 'HF/P', 'ULF', 'VHF']

features_names_freqhvr = ['ULF', 'VLF', 'LF', 'HF', 'VHF', 'Total', 'extra1', 'extra2', 
                            'extra3', 'extra4', 'extra5', 'corr_max', 'corr_min', 'corr1', 'corr2', 'corr3']


for i in range(numberOfTestSamples):
    if i % 500 == 0:
        print(i)

    # Create an empty arrray, features will be appended to this
    features = np.empty([1,0])
    # Get the current signal
    currentPatient = X_train.iloc[i].dropna().values
    # preproces the signal, it returns a dictionary
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(currentPatient, sampling_rate=300, show=False)
    templates, rpeaks_ = ecg.extract_heartbeats(filtered, rpeaks = rpeaks, sampling_rate=300)

    # if heart_rate.size == 0:
    #     print('array is empty at index: ', i)

    # add mean, max, min, median, std of r amplitudes
    R_Peaks = filtered[rpeaks]

    mean_rvalues = np.mean(R_Peaks)
    min_rvalues = np.min(R_Peaks)
    max_rvalues = np.max(R_Peaks)
    std_rvalues = np.std(R_Peaks)
    median_rvalues = np.median(R_Peaks)
    
    means_of_r = np.empty([1, 5])
    means_of_r[0, :] = [mean_rvalues, min_rvalues, max_rvalues, std_rvalues, median_rvalues]
    features = np.append(features, means_of_r, axis=1)

    # add mean, max, min, median, std of heart rates
    # mean_hr = np.mean(heart_rate)
    # max_hr = np.mean(heart_rate)
    # min_hr = np.mean(heart_rate)
    # median_hr = np.mean(heart_rate)
    # std_hr = np.mean(heart_rate)

    # heart_rate_features = np.empty([1,5])
    # heart_rate_features[0,:] = [mean_hr, max_hr, min_hr, median_hr, std_hr]
    # features = np.append(features, heart_rate_features, axis=1)

    # add power
    power = np.sum(np.square(filtered)) / filtered.shape[0]
    features = np.append(features, power.reshape(1,-1), axis=1)



    Cardiadic_Cycles = pd.DataFrame(templates)

    
    # add the mean, mean-max, mean-min, mean-std, mean-median of cardiac cycles
    mean = Cardiadic_Cycles.mean(axis=0).mean()
    mean_max = Cardiadic_Cycles.max(axis=0).mean()
    mean_min = Cardiadic_Cycles.min(axis=0).mean()
    mean_median = Cardiadic_Cycles.median(axis=0).mean()
    mean_std = Cardiadic_Cycles.std(axis=0).mean()

    max_min = Cardiadic_Cycles.min(axis=0).max()
    min_min = Cardiadic_Cycles.min(axis=0).min()
    
    max_max = Cardiadic_Cycles.max(axis=0).max()
    min_max = Cardiadic_Cycles.max(axis=0).min()

    max_std = Cardiadic_Cycles.std(axis=0).max()
    min_std = Cardiadic_Cycles.std(axis=0).min()

    to_add = np.empty([1,11])
    to_add[0, :] = [mean, mean_max, mean_min, mean_median, mean_std, max_min, min_min, max_max, min_max, max_std, min_std]

    features= np.append(features, to_add, axis=1)    


    # Addition of time hvr features
    hvr_time_features = nkcopy.ecg_hrv(rpeaks=rpeaks, sampling_rate=300, hrv_features='time')
    # add all time hvr to array features
    for feature in features_names_timehvr:
        features = np.append(features, hvr_time_features[feature])
    
    # Addition of frequency features
    hvr_freq_features = frequency_analysis.get_frequency_features(templates)
    # add all frequency hvr to array features
    for feature in features_names_freqhvr:
        features = np.append(features, hvr_freq_features[feature])
    
    preprocessed_X_train[i,:] = features



np.save('preprocessed_X_test_43', preprocessed_X_train)

