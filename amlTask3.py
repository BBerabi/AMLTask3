import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biosppy.signals.ecg as ecg
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import auxilary

FEATURE_SELECTION = False
REMOVE_OUTLIER = True
PLOT_FEATURE_SCORES = False


X_train = np.load('preprocessed_X_train_20.npy')
X_test = np.load('preprocessed_X_test_20.npy')

print(X_train.shape)
print(X_test.shape)

y_train = pd.read_csv(r'y_train.csv')
y_train = y_train.drop(columns= 'id', axis=1)
y_train = y_train.values.ravel()

print("train shape: ", X_train.shape)
print("test shape: ", X_test.shape)

# Remove Outliers
if REMOVE_OUTLIER:
    X_train, y_train = auxilary.OutlierDetectionIsolationForest(X_train, y_train, percentageOutlier='auto')
# Plot scores of features
if PLOT_FEATURE_SCORES:
    auxilary.plotSFeatureScores(X_train, y_train, f_classif)
# Feature Selection !
if FEATURE_SELECTION:
    inputDim = 120
    featureSelection = SelectKBest(f_classif, k = inputDim)
    X_train = featureSelection.fit_transform(X_train, y_train)
    scores = featureSelection.scores_
    print("Shape after feature selection: ", X_train.shape)

normalizer = Normalizer()
scaler = StandardScaler()

#X_train = scaler.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)

# First train a binary classifier on labels [0,1,2] vs 3

y_train_BinaryClassifier = np.copy(y_train)

indicesLabel1 = np.argwhere(y_train_BinaryClassifier == 1)
indicesLabel2 = np.argwhere(y_train_BinaryClassifier == 2)
y_train_BinaryClassifier[indicesLabel1] = 0
y_train_BinaryClassifier[indicesLabel2] = 0

indicesLabel3 = np.argwhere(y_train_BinaryClassifier == 3)
y_train_BinaryClassifier[indicesLabel3] = 1

print('Label 1: ', indicesLabel1.shape)
print('Label 2: ', indicesLabel2.shape)
print('Label 3: ', indicesLabel3.shape)

parameters_BinaryClassifier = { 'n_estimators': [10, 100, 250, 500, 1000],
                    'criterion': ['entropy'],
                    'class_weight': ['balanced']
                 }
rfc_bc = RandomForestClassifier(random_state=0)


parameters_BinaryClassifier = {
    'kernel': ['rbf', 'linear'],
    'C': [0.001, 0.01,0.1,1,10]
}
svc = SVC(gamma='scale', class_weight='balanced', random_state=37, decision_function_shape='ovo')




scoreFunction = make_scorer(f1_score, average='micro', greater_is_better=True)

clf_bc = GridSearchCV(estimator=svc, param_grid=parameters_BinaryClassifier, cv=5, verbose=2, scoring=scoreFunction, n_jobs=3)
clf_bc.fit(X_train, y_train_BinaryClassifier)

print("Best score of best on validation set: ", clf_bc.best_score_) #rbf, 1e-5
print("Best Parameters: ", clf_bc.best_params_) # 0.968448

#X_test = scaler.transform(X_test)
X_test = normalizer.transform(X_test)
y_pred_bc = clf_bc.predict(X_test)

numberOfOthers = np.count_nonzero(y_pred_bc == 0)
numberOfLabel3 = np.count_nonzero(y_pred_bc == 1)


print('prediction others:', numberOfOthers)

print('prediction label3:', numberOfLabel3)

print('Number of 0:', np.count_nonzero(y_train == 0))
print('Number of 1:', np.count_nonzero(y_train == 1))
print('Number of 2:', np.count_nonzero(y_train == 2))
print('Number of 3:', np.count_nonzero(y_train == 3))


# Now remove the class 3 from data and train a second classifier

X_train = np.delete(X_train, indicesLabel3, axis=0)
y_train = np.delete(y_train, indicesLabel3, axis=0)

print('Number of 0:', np.count_nonzero(y_train == 0))
print('Number of 1:', np.count_nonzero(y_train == 1))
print('Number of 2:', np.count_nonzero(y_train == 2))
print('Number of 3:', np.count_nonzero(y_train == 3))



parameters_rfc = { 'n_estimators': [10, 100, 250, 500, 1000],
                    'criterion': ['entropy'],
                    'class_weight': ['balanced']
                 }
rfc = RandomForestClassifier(random_state=0)


clf = GridSearchCV(estimator=rfc, param_grid= parameters_rfc, cv=5, verbose=2, scoring= scoreFunction, n_jobs=3)
clf.fit(X_train, y_train)


print("Best score of best on validation set: ", clf.best_score_) #0.6670
print("Best Parameters: ", clf.best_params_) #rbf, 10


# First predict the first 3 classes

X_test = scaler.transform(X_test)

y_pred_test = clf.predict(X_test)

y_pred_test_bc = clf_bc.predict(X_test)
indicesOfLabel3Prediction = np.argwhere(y_pred_test_bc == 1)

y_pred_test[indicesOfLabel3Prediction] = 3
auxilary.createSubmissionFiles(y_pred_test)














######## Plot some samples #########

# for i in range(10):
#     randomIndex = np.random.randint(0, 5117)    
#     sample = X_train[randomIndex,:]
#     sample = pd.Series(sample)
#     sample.plot()
#     label = y_train[randomIndex]
#     plt.title('y index: %i' % label )
#     plt.show(block=False)
#     plt.pause(1)
#     plt.close()

# indicesOfLabel3 = np.where(y_train == 3)[0]
# print(indicesOfLabel3.shape)
# for i in range(10):
#     randomIndex = np.random.randint(0, indicesOfLabel3.shape[0])    
#     sample = X_train[indicesOfLabel3[randomIndex],:]
#     sample = pd.Series(sample)
#     sample.plot()
#     label = y_train[indicesOfLabel3[randomIndex]]
#     plt.title('y index: %i' % label )
#     plt.show(block=False)
#     plt.pause(1)
#     plt.close()




