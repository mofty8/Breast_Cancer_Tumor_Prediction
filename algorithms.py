import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data.csv')
data.shape
data.drop(data.columns[32], axis=1, inplace=True)
ids = data['id']
data.drop(['id'], axis=1, inplace=True)

scaler = StandardScaler()
data.iloc[:,1:] = scaler.fit_transform(data.iloc[:,1:])

X = data.iloc[:,1:]
y = data.diagnosis

#print('original data shapes:', X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#print('splitted data shapes:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
models = []
models.append(('K-Nearest Neighbours (4)', KNeighborsClassifier(n_neighbors=4)))
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Gaussian Naïve Bayes', GaussianNB()))

names = []
scores = []
falnegs = []

best_model = None
highest_score = 0.0
false_negatives = None

for name, model in models:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_pred, y_test)
    tn, fp, fn, tp = cm.ravel()
    print(name, '\n', cm, '\n')
    print(score)

    names.append(name)
    scores.append(score)
    falnegs.append(fn)

    if ((score > highest_score) or (score == highest_score and fn < false_negatives)):
        best_model = model
        highest_score = score
        false_negatives = fn

print('Best model:', best_model)




