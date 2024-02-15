# classify-iris-simple.py
# parsons/29-jan-2017
#
# Using a decision tree on the iris dataset.
#
# Code is based on:
#
# http://scikit-learn.org/stable/modules/tree.html)
# http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


# Parameters
plot_step = 0.02
num_trees = 10

# Score table
scores = []

# Load the data
bc = load_breast_cancer()

# 10 random trees
for i in range(num_trees):

    X_train, X_test, y_train, y_test = train_test_split(
    bc.data [:, [2, 6]], bc.target, test_size=0.1, random_state=0) # features 2 and 6

    # Now create a decision tree and fit it to the iris data:
    bc_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)

    # Now plot the decision surface that we just learnt by using the decision tree to
    # classify every packground point.
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                        np.arange(y_min, y_max, plot_step))

    Z = bc_tree.predict(np.c_[xx.ravel(), yy.ravel()]) # Here we use the tree
                                                        # to predict the classification
                                                        # of each background point.
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    # Also plot the original data on the same axes
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.astype(float))#, cmap='autumn')

    # Label axes
    plt.xlabel( bc.feature_names[2], fontsize=10 )
    plt.ylabel( bc.feature_names[6], fontsize=10 )

    scores.append(bc_tree.score(X_test, y_test))

    plt.show()

print("All scores: " + str(scores))
print("Mean value: " + str(np.mean(scores)))
print("Standard error: " + str(np.std(scores)))

# Cross-validation 
X_train, X_test, y_train, y_test = train_test_split(
    bc.data [:, [2, 6]], bc.target, test_size=0.1, random_state=0) # features 2 and 6
bc_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)
cv_scores = cross_val_score(bc_tree, X_test, y_test, cv=10)

print("Cross-validation scores: " + str(cv_scores))
print("Mean value: " + str(np.mean(cv_scores)))
print("Standard error: " + str(np.std(cv_scores)))

# Metrics

X_train, X_test, y_train, y_test = train_test_split(
    bc.data [:, [2, 6]], bc.target, test_size=0.1, random_state=0) # features 2 and 6
yr_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)

# Confustion matrix
TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(X_test)):
    X = X_test[i,[0, 1]]
    y_predict = yr_tree.predict(X.reshape(1, -1))
    y_true = y_test[i]

    if y_predict == 1 and y_true == 1:
        TP = TP + 1
    elif y_predict == 0 and y_true == 0:
        TN = TN + 1
    elif y_predict == 1 and y_true == 0:
        FP = FP + 1
    else: 
        FN = FN + 1

print("TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN))

# Precision
p = TP/(TP+FP)
print("Precision: " + str(p))

# Recall
r = TP/(TP+FN)
print("Recall: " + str(r))

# F1 score
f1 = 2*((p*r)/(p+r))
print("F1 score: " + str(f1))

# ROC curve
my_neigh = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

prob_TP = 0
prob_FP = 0
for i in range(len(X_test)):
    X = X_test[i,[0, 1]]
    y_predict = my_neigh.predict_proba(X.reshape(1, -1))

    if y_predict > 0.8:
        prob_TP = prob_TP + 1
    else:
        prob_FP = prob_FP + 1

