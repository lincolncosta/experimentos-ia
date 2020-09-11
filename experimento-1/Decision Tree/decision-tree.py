import pandas as pd
# Import train_test_split function
from sklearn.model_selection import train_test_split, cross_val_score
# Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import graphviz
import matplotlib.pyplot as plt
import csv

dataset = pd.read_csv("../data/banana.csv")
X = dataset.drop('classe', axis=1)
y = dataset['classe']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X, y)
y_pred = classifier.predict(X_test)

confusionMatrix = pd.crosstab(y_test, y_pred)
classificationReport = classification_report(
    y_test, y_pred, output_dict=True)
scoresCV2 = cross_val_score(classifier, X, y, cv=2)
scoresCV5 = cross_val_score(classifier, X, y, cv=5)
scoresCV10 = cross_val_score(classifier, X, y, cv=10)

confusionMatrix.to_csv('confusion-matrix-dt.csv')
pd.DataFrame(classificationReport).transpose().to_csv(
    'classification-report-dt.csv')

with open('cross-validation-dt.csv', mode='a') as CVFile:
    writer = csv.writer(CVFile)
    writer.writerow([2, scoresCV2, scoresCV2.mean()])
    writer.writerow([5, scoresCV5, scoresCV5.mean()])
    writer.writerow([10, scoresCV10, scoresCV10.mean()])


plt.figure()
plot_tree(classifier, filled=True)
plt.savefig('decision-tree.png')
