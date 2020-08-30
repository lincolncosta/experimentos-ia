import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def generateScatter(dataset):
    color_dict = dict({1.0: 'red',
                       -1.0: 'dodgerblue'})
    scatter = sns.scatterplot(x="at1", y="at2", hue="classe",
                              data=dataset, palette=color_dict)
    scatter.set(xlabel='Atributo 1', ylabel='Atributo 2')
    return plt.show()


def setupDataset(dataset):
    X = dataset.drop('classe', axis=1)
    y = dataset['classe']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    return X_train, X_test, y_train, y_test


def applyKernel(kernelType, X_train, X_test, y_train, coef0=0.0):
    classifier = SVC(kernel=kernelType, coef0=coef0)
    classifier.fit(X_train, y_train)
    predY = classifier.predict(X_test)
    return predY


dataset = pd.read_csv("../data/banana.csv")
X_train, X_test, y_train, y_test = setupDataset(dataset)

predSigmoidY1 = applyKernel('sigmoid', X_train, X_test, y_train, 1)
predSigmoidY05 = applyKernel('sigmoid', X_train, X_test, y_train, 0.5)
predSigmoidY001 = applyKernel('sigmoid', X_train, X_test, y_train, 0.01)
predLinearY = applyKernel('linear', X_train, X_test, y_train)
predPolyY = applyKernel('poly', X_train, X_test, y_train)
predRBFY = applyKernel('rbf', X_train, X_test, y_train)
