import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("../data/banana.csv")


def generateScatter(dataset):
    color_dict = dict({1.0: 'red',
                       -1.0: 'dodgerblue'})
    scatter = sns.scatterplot(x="at1", y="at2", hue="classe",
                              data=dataset, palette=color_dict)
    scatter.set(xlabel='Atributo 1', ylabel='Atributo 2')
    return plt.show()

def setupDataset():
    X = dataset.drop('classe', axis=1)
    y = dataset['classe']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    return X_train, X_test, y_train, y_test


sigmoidClassifier = SVC(kernel='sigmoid')
sigmoidClassifier.fit(X_train, y_train)
predSigmoidY = sigmoidClassifier.predict(X_test)

print(confusion_matrix(y_test, predSigmoidY))
print(classification_report(y_test, predSigmoidY))

linearClassifier = SVC(kernel='linear')
linearClassifier.fit(X_train, y_train)
predLinearY = linearClassifier.predict(X_test)

print(confusion_matrix(y_test, predSigmoidY))
print(classification_report(y_test, predLinearY))

polyClassifier = SVC(kernel='poly', degree=8)
polyClassifier.fit(X_train, y_train)
predPolyY = polyClassifier.predict(X_test)

print(confusion_matrix(y_test, predPolyY))
print(classification_report(y_test, predPolyY))

RBFClassifier = SVC(kernel='rbf')
RBFClassifier.fit(X_train, y_train)
predRBFY = RBFClassifier.predict(X_test)

print(confusion_matrix(y_test, predRBFY))
print(classification_report(y_test, predRBFY))
