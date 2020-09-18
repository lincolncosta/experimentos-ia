import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import seaborn as sns
import csv
sns.set()


def main():

    dataset, X, y, X_train, X_test, y_train, y_test = setupDataset()

    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train, y_train)
    predY = knn.predict(X_test)

    accuracy = accuracy_score(y_test, predY)
    print(accuracy)
    evaluateKNN(X, y, X_test, y_test, predY, 'KNN', knn)
    generateGraphs(X.to_numpy(), y.to_numpy(dtype=np.int64), knn, dataset)


def setupDataset():
    dataset = pd.read_csv("../data/banana.csv")
    X = dataset.drop('Classe', axis=1)
    y = dataset['Classe']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    return dataset, X, y, X_train, X_test, y_train, y_test


def evaluateKNN(X, y, X_test, y_test, y_pred, kernelType, classifier):
    confusionMatrix = pd.crosstab(y_test, y_pred)
    classificationReport = classification_report(
        y_test, y_pred, output_dict=True)
    confusionMatrix.to_csv('confusion-matrix-{}.csv'.format(kernelType))
    pd.DataFrame(classificationReport).transpose().to_csv(
        'classification-report-{}.csv'.format(kernelType))

    kfold2 = KFold(n_splits=2)
    scoresCV2 = cross_val_score(classifier, X, y, cv=kfold2)
    kfold5 = KFold(n_splits=5)
    scoresCV5 = cross_val_score(classifier, X, y, cv=kfold5)
    kfold10 = KFold(n_splits=10)
    scoresCV10 = cross_val_score(classifier, X, y, cv=kfold10)

    with open('cross-validation-file.csv', mode='a') as CVFile:
        writer = csv.writer(CVFile)
        writer.writerow([kernelType, 2, scoresCV2,
                         scoresCV2.mean(), 1 - scoresCV2.mean()])
        writer.writerow([kernelType, 5, scoresCV5,
                         scoresCV5.mean(), 1 - scoresCV5.mean()])
        writer.writerow([kernelType, 10, scoresCV10,
                         scoresCV10.mean(), 1 - scoresCV10.mean()])


def generateGraphs(X, y, classifier, dataset, h=0.2):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.clf()

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    color_dict = dict({1.0: 'red',
                       -1.0: 'dodgerblue'})
    scatter = sns.scatterplot(x="at1", y="at2", hue="Classe",
                                data=dataset, palette=color_dict)
    scatter.set(xlabel=None, ylabel=None)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.savefig('knn.png')


main()
