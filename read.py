import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os

data = pd.read_csv("data.csv")
#data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
#data.tail()
#data.info()
data.drop(["id","Unnamed: 32"],axis = 1,inplace = True)
data.info()

M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

plt.scatter(M.radius_mean,M.area_mean,color = "Black",label="Malignant",alpha=0.2)
plt.scatter(B.radius_mean,B.area_mean,color = "Red",label="Benign",alpha=0.3)
plt.xlabel("Radius Mean")
plt.ylabel("Area Mean")
plt.legend()
plt.show()

plt.scatter(M.radius_mean,M.symmetry_worst,color = "Black",label="Malignant",alpha=0.2)
plt.scatter(B.radius_mean,B.symmetry_worst,color = "Red",label="Benign",alpha=0.3)
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.legend()
plt.show()

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y= data.diagnosis.values
x1 = data.drop(["diagnosis"],axis= 1)
print(y)

x = (x1-np.min(x1))/(np.max(x1)-np.min(x1))

xtrain, xtest, ytrain, ytest =  train_test_split(x,y,test_size=0.3,random_state=42)

k = 4
KNN = KNeighborsClassifier(n_neighbors =k )
KNN.fit(xtrain,ytrain)
prediction = KNN.predict(xtest)

print("{}-NN Score: {}".format(k,KNN.score(xtest,ytest)))

scores = []
for each in range(1, 14):
    KNNfind = KNeighborsClassifier(n_neighbors=each)
    KNNfind.fit(xtrain, ytrain)
    scores.append(KNNfind.score(xtest, ytest))

plt.plot(range(1, 14), scores, color="black")
plt.xlabel("K Values")
plt.ylabel("Score(Accuracy)")
plt.show()













