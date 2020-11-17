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
import xgboost as xgb

OUTLIER_DETECTION = False
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

if OUTLIER_DETECTION:
    X_train, y_train = auxilary.OutlierDetectionIsolationForest(X_train, y_train, percentageOutlier = 0.002)


auxilary.plotSFeatureScores(X_train, y_train, f_classif)
if FEATURE_SELECTION:
    inputDim = 40
    featureSelection = SelectKBest(f_classif, k = inputDim)
    X_train = featureSelection.fit_transform(X_train, y_train)
    scores = featureSelection.scores_
    print("Shape after feature selection: ", X_train.shape)




scoreFunction = make_scorer(f1_score, average='micro', greater_is_better=True)

xgb_model = xgb.XGBClassifier()

parameters = {
    'objective': ['binary:logistic'],
    'max_depth': [10],
    'min_child_weight': [11],
    'n_estimators': [400],
    'seed': [1111],
    'learning_rate': [0.05],
    'max_delta_step': [3],
    'num_class': [4]
}

clf = GridSearchCV(estimator=xgb_model, param_grid=parameters, n_jobs=5, cv=10, scoring=scoreFunction, verbose=2)
clf.fit(X_train, y_train)



print("Best score of best on validation set: ", clf.best_score_) #0.6670
print("Best Parameters: ", clf.best_params_) #rbf, 10


X_test = scaler.transform(X_test)
if FEATURE_SELECTION:
   X_test = featureSelection.transform(X_test)
y_pred_test = clf.predict(X_test)
print('Number of 3:', np.count_nonzero(y_pred_test == 3))

auxilary.createSubmissionFiles(y_pred_test)
