#--
# classify-iris.py
# sklar/22-jan-2017
# This code demonstrates Decision Tree classification on the Iris data set, using the scikit-learn toolkit.
# Source is based on code from:
#  http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html
#--

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

#--
# define parameters
#--
n_classes = 2
plot_colors = "bry"
plot_step = 0.02


#--
# load data 
#--
#iris = load_iris()

bc = load_breast_cancer()
bc_df = pd.DataFrame(bc.data, columns=bc.feature_names)
bc_df['target'] = pd.Series(bc.target)
bc_df.drop('mean area', axis=1, inplace=True) #error with column
num_features = 9

X_train, X_test, y_train, y_test = train_test_split(
    bc_df.loc[:, 'mean radius' : 'mean fractal dimension'], bc_df.loc[:, 'target'], test_size=0.1, random_state=0) #only first 9 features



#--
# classify and plot data
#--
plt.figure()
plt.rc( 'xtick', labelsize=8 )
plt.rc( 'ytick', labelsize=8 )
for i in range(0,num_features):
    for j in range(i+1,num_features):
        # classify using two corresponding features
        pair = [i, j]
        X = X_train.iloc[:, pair]
        y = y_train
        # train classifier
        clf = DecisionTreeClassifier().fit( X,  y )
        # plot the (learned) decision boundaries
        plt.subplot( num_features, num_features, j*num_features+i+1 )
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid( np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step) )
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
        plt.xlabel(X.columns[0], fontsize=8 )
        plt.ylabel(X.columns[1], fontsize=8 )
        plt.axis( "tight" )
        # plot the training points
        for ii, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == ii)
            plt.scatter(X.iloc[idx[0], 0], X.iloc[idx[0], 1], c=color, label=bc.target_names[ii],cmap=plt.cm.Paired)
        plt.axis("tight")
        print('Accuracy for ' + str(X.columns.values))
        print(clf.score(X_test.iloc[:, pair], y_test))
plt.show()


