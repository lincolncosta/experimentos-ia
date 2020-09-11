import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import csv

titles = ['Sigmoid coef 1',
          'Sigmoid coef 0.5',
          'Sigmoid coef 0.01',
          'Sigmoid Linear',
          'Sigmoid Poly',
          'Sigmoid RBF']


def setupDataset(dataset):
    X = dataset.drop('classe', axis=1)
    y = dataset['classe']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    return X, y, X_train, X_test, y_train, y_test


def applyKernel(kernelType, X_train, X_test, y_train, coef0=0.0):
    classifier = SVC(kernel=kernelType, coef0=coef0)
    classifier.fit(X_train, y_train)
    predY = classifier.predict(X_test)
    return classifier, predY


def evaluateKernel(y_test, y_pred, kernelType, classifier):
    confusionMatrix = pd.crosstab(y_test, y_pred)
    classificationReport = classification_report(
        y_test, y_pred, output_dict=True)
    confusionMatrix.to_csv('confusion-matrix-{}.csv'.format(kernelType))
    pd.DataFrame(classificationReport).transpose().to_csv(
        'classification-report-{}.csv'.format(kernelType))

    scoresCV2 = cross_val_score(classifier, X, y, cv=2)
    scoresCV5 = cross_val_score(classifier, X, y, cv=5)
    scoresCV10 = cross_val_score(classifier, X, y, cv=10)

    with open('cross-validation-file.csv', mode='a') as CVFile:
        writer = csv.writer(CVFile)
        writer.writerow([kernelType, 2, scoresCV2, scoresCV2.mean()])
        writer.writerow([kernelType, 5, scoresCV5, scoresCV5.mean()])
        writer.writerow([kernelType, 10, scoresCV10, scoresCV10.mean()])


def generateGraphs(X, y, classifiers, h=0.2):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    for i, classifier in classifiers:
        plt.subplot(3, 3, i + 1)
        plt.subplots_adjust(wspace=0.6, hspace=0.6)

        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])
    plt.savefig('kernel-types.png')


dataset = pd.read_csv("../data/banana.csv")
X, y, X_train, X_test, y_train, y_test = setupDataset(dataset)

classifierSigmoid1, predSigmoidY1 = applyKernel(
    'sigmoid', X_train, X_test, y_train, 1)
evaluateKernel(y_test, predSigmoidY1, 'sigmoid1', classifierSigmoid1)

classifierSigmoid05, predSigmoidY05 = applyKernel(
    'sigmoid', X_train, X_test, y_train, 0.5)
evaluateKernel(y_test, predSigmoidY05, 'sigmoid05', classifierSigmoid05)

classifierSigmoid001, predSigmoidY001 = applyKernel(
    'sigmoid', X_train, X_test, y_train, 0.01)
evaluateKernel(y_test, predSigmoidY001, 'sigmoid001', classifierSigmoid001)

classifierLinear, predLinearY = applyKernel('linear', X_train, X_test, y_train)
evaluateKernel(y_test, predLinearY, 'linear', classifierLinear)

classifierPoly, predPolyY = applyKernel('poly', X_train, X_test, y_train)
evaluateKernel(y_test, predPolyY, 'poly', classifierPoly)

classifierRBF, predRBFY = applyKernel('rbf', X_train, X_test, y_train)
evaluateKernel(y_test, predRBFY, 'RBF', classifierRBF)

plt = generateGraphs(X.to_numpy(), y.to_numpy(dtype=np.int64), enumerate((classifierSigmoid1,
                                                                          classifierSigmoid05, classifierSigmoid001, classifierLinear, classifierPoly, classifierRBF)))
