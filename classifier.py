import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biosppy.signals.ecg as ecg
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import neurokit as nk
import auxilary

FEATURE_SELECTION = True

X_train = np.load('preprocessed_X_train_43.npy')
X_test = np.load('preprocessed_X_test_43.npy')

print(X_train.shape)
print(X_test.shape)

y_train = pd.read_csv(r'y_train.csv')
y_train = y_train.drop(columns= 'id', axis=1)
y_train = y_train.values.ravel()


print("train shape: ", X_train.shape)
print("test shape: ", X_test.shape)

scaler = StandardScaler()
#normalizer = Normalizer()
#X_train = normalizer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)

X_train, y_train = auxilary.OutlierDetectionIsolationForest(X_train, y_train, percentageOutlier = 0.001)


auxilary.plotSFeatureScores(X_train, y_train, f_classif)
if FEATURE_SELECTION:
    inputDim = 40
    featureSelection = SelectKBest(f_classif, k = inputDim)
    X_train = featureSelection.fit_transform(X_train, y_train)
    scores = featureSelection.scores_
    print("Shape after feature selection: ", X_train.shape)




scoreFunction = make_scorer(f1_score, average='micro', greater_is_better=True)
parameters = {  'kernel': ['rbf', 'linear', 'poly'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100]

                }

parameters_rfc = { 'n_estimators': [1100],
                    'criterion': ['entropy'],
                    'class_weight': ['balanced']
                 }

parameters_gbc = { 'learning_rate': [0.1, 0.01],
                  'loss': ['deviance'],
                  'n_estimators': [3000]
}



gbc = GradientBoostingClassifier(random_state=24)
            

rfc = RandomForestClassifier(random_state=0)

svc = SVC(gamma = 'scale', random_state=27, max_iter=50000,
            decision_function_shape= 'ovo', degree=3)


clf = GridSearchCV(estimator=rfc, param_grid= parameters_rfc, cv=10, verbose=2, scoring= scoreFunction, n_jobs=3)
clf.fit(X_train, y_train)

print("Best score of best on validation set: ", clf.best_score_) #0.6670
print("Best Parameters: ", clf.best_params_) #rbf, 10


X_test = scaler.transform(X_test)
if FEATURE_SELECTION:
   X_test = featureSelection.transform(X_test)
y_pred_test = clf.predict(X_test)
print('Number of 3:', np.count_nonzero(y_pred_test == 3))

auxilary.createSubmissionFiles(y_pred_test)
