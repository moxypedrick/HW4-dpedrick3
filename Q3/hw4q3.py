## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect eye state

import sklearn
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.decomposition import PCA


######################################### Reading and Splitting the Data ###############################################
# XXX
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
# XXX

data = pd.read_csv('C:/Users/dpedrick/OneDrive/GaTech/Data_and_Visual_Analytics/HW4-dpedrick3/hw4-skeleton/Q3/eeg_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]


# The random state to use while splitting the data.
random_state = 100

# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
# XXX
x_train,x_test = train_test_split(x_data, test_size = 0.3, shuffle = True, random_state=100)
y_train,y_test = train_test_split(y_data, test_size = 0.3, shuffle = True, random_state=100)

# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# XXX

lnrRgr = linear_model.LinearRegression()
lnrRgr.fit(x_train,y_train)


# XXX
# TODO: Test its accuracy (on the training set) using the accuracy_score method.
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use y_predict.round() or any other method.
# XXX

y_pred = lnrRgr.predict(x_train)
y_pred2 = lnrRgr.predict(x_test)

print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred2))


# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
# X


clf = RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)

clf.fit(x_train,y_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

y_pred3_train = clf.predict(x_train)
y_pred3_test = clf.predict(x_test)

print(accuracy_score(y_train, y_pred3_train))
print(accuracy_score(y_test, y_pred3_test))

# XXX
# TODO: Determine the feature importance as evaluated by the Random Forest Classifier.
#       Sort them in the descending order and print the feature numbers. The report the most important and the least important feature.
#       Mention the features with the exact names, e.g. X11, X1, etc.
#       Hint: There is a direct function available in sklearn to achieve this. Also checkout argsort() function in Python.
# XXX

coefs_list = clf.feature_importances_
col_names = data.columns.values
coef_sorted_list = np.argsort(coefs_list)
coef_sorted_list2 = []
for i in range(len(data.columns.values)-1):
    coef_sorted_list2.append(col_names[coef_sorted_list[i]])

print(coef_sorted_list2)
# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX


clf2 = RandomForestClassifier(random_state=0)

n_estimators_list = np.array([10,100])
max_depth_list = np.array([2,4,8])

grid = GridSearchCV(estimator=clf2, cv=10, param_grid=dict(n_estimators=n_estimators_list, max_depth = max_depth_list))

grid.fit(x_train,y_train)

print(grid.best_score_)
print(grid.best_estimator_.n_estimators)
print(grid.best_estimator_.max_depth)
print(grid.best_params_)


# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
# TODO: Create a SVC classifier and train it.
# XXX

x_normalized = sklearn.preprocessing.normalize(x_data, norm="l2")

x_train_norm,x_test_norm = train_test_split(x_normalized, test_size = 0.3, shuffle = True, random_state=100)
clf3 = SVC(gamma = "auto", C = 1.0,  kernel = "rbf")
clf3.fit(x_train_norm, y_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX
y_pred4_train = clf3.predict(x_train_norm)
y_pred4_test = clf3.predict(x_test_norm)

print(accuracy_score(y_test, y_pred4_test))
print(accuracy_score(y_train, y_pred4_train))

# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX

c_list = np.array([0.01, 1])
kernel_list = np.array(["rbf","linear"])
clf4 = SVC(gamma="auto")
grid3 = GridSearchCV(estimator = clf4, cv = 10, param_grid = dict(C = c_list, kernel = kernel_list))
grid3.fit(x_train_norm, y_train)

y_pred4_train_grid = grid3.predict(x_train_norm)
y_pred4_test_grid = grid3.predict(x_test_norm)


print(accuracy_score(y_test, y_pred4_test_grid))
print(accuracy_score(y_train, y_pred4_train_grid))

print(grid3.best_score_)
print(grid3.best_params_)

# ######################################### Principal Component Analysis #################################################
# XXX
# TODO: Perform dimensionality reduction of the data using PCA.
#       Set parameters n_component to 10 and svd_solver to 'full'. Keep other parameters at their default value.
#       Print the following arrays:
#       - Percentage of variance explained by each of the selected components
#       - The singular values corresponding to each of the selected components.
# XXX

pca = PCA(n_components=10, svd_solver='full')
pca.fit(x_test)

print(pca.explained_variance_)
print(pca.singular_values_)