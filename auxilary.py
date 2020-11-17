import numpy as np
import numpy.linalg as LA
from scipy import linalg as LA2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import biosppy.signals.ecg as ecg
import random

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_classif


def plotSFeatureScores(X_train, y_train, score_function):
    inputDim = 20
    featureSelection = SelectKBest(score_function, k = inputDim)
    featureSelection.fit(X_train, y_train)
    scores = featureSelection.scores_
    scores = np.sort(scores)[::-1]
    axis = np.arange(1,X_train.shape[1]+1)
    scores = np.nan_to_num(scores, nan = 0)
    print('scores: ', scores)
    total = np.sum(scores)
    #print("total: ", total)
    plt.plot(axis, scores, 'ro',marker = '+', markersize = 0.4)
    #plt.plot(axis, scores/total, 'ro', marker = '+', markersize = 0.4)
    plt.show()


def OutlierDetectionIsolationForest(data, labels, percentageOutlier):
    
    clf = IsolationForest( behaviour = 'new', max_samples=0.99, random_state = 1, contamination= percentageOutlier)
    preds = clf.fit_predict(data)

    indicesToRemove = np.argwhere(preds == -1)
    numberOfOutliers = np.count_nonzero(preds == -1)
    print("Number Of Outliers:", numberOfOutliers)
    data = np.delete(data, indicesToRemove, axis = 0)
    labels = np.delete(labels, indicesToRemove)

    return data, labels


def createSubmissionFiles(y_predictions):
    output = pd.DataFrame()
    output.insert(0, 'y', y_predictions)
    #A = pd.read_csv(r"C:\Users\berka\Desktop\task3\X_test.csv")
    output.index = np.arange(0, 3411) #A.index
    output.index.names = ['id']
    output.to_csv("output")
    print("Submission files are succesfully created")



######### Extract into heartbeats and calculate the average for all patients #########
######### Save into an npy file and read there for further trials ############
def aggregateHeartbeats(data, filename):        
    print("Started with aggregation")

    numberOfPatients = data.shape[0]
    averagedHeartBeats = np.empty([numberOfPatients, 180])

    for i in range(numberOfPatients):
        currentPatient = data.iloc[i]
        currentPatient = currentPatient.dropna()
        currentPatient = currentPatient.values
        #rpeaks_ = ecg.correct_rpeaks(currentPatient, )
        ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(currentPatient, sampling_rate=300, show=False)
        
        #rpeaks_ = ecg.hamilton_segmenter(filtered, sampling_rate=300)
        #rpeaks_ = ecg.gamboa_segmenter(currentPatient, rpeaks_, sampling_rate=300)
        #templates, rpeaks_ = ecg.extract_heartbeats(currentPatient, rpeaks = rpeaks_, sampling_rate=300)
        print("shape before summing out:", templates.shape)
        if templates.shape[1] != 180:
            print("anasini sikim boyle datanin")
        templates = np.sum(templates, axis=0) / templates.shape[0]
        print("shape after summing out:", templates.shape)
        averagedHeartBeats[i] = templates

    print("Shape of averaged heartbeats: ", averagedHeartBeats.shape)
    np.save(filename, averagedHeartBeats)

