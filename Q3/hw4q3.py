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

data = pd.read_csv('C:/Users/dpedrick/OneDrive/GaTech/Data_and_Visual_Analytics/HW4-dpedrick3/hw4-skeleton/Q3/eeg_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100

# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.

x_train,x_test = train_test_split(x_data, test_size = 0.3, shuffle = True, random_state=100)
y_train,y_test = train_test_split(y_data, test_size = 0.3, shuffle = True, random_state=100)

# ############################################### Linear Regression ###################################################

lnrRgr = linear_model.LinearRegression()
lnrRgr.fit(x_train,y_train)

y_pred_train = lnrRgr.predict(x_train).round()
y_pred_test = lnrRgr.predict(x_test).round()

#print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred2))
print('Linear Regression Accuracy: ')
print('Linear Regression - Training: %.2f' % accuracy_score(y_train, y_pred_train))
print('Linear Regression - Test: %.2f' %accuracy_score(y_test, y_pred_test))


################################################ Random Forest Classifier ##############################################

clf = RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)
clf.fit(x_train,y_train)

y_pred2_train = clf.predict(x_train)
y_pred2_test = clf.predict(x_test)

print('Random Forest Accuracy: ')
print('Random Forest - Training: %.2f' % accuracy_score(y_train, y_pred2_train))
print('Random Forest - Training: %.2f' % accuracy_score(y_test, y_pred2_test))

######################################## SVM ###############################################
x_normalized = sklearn.preprocessing.normalize(x_data, norm="l2")

x_train_norm,x_test_norm = train_test_split(x_normalized, test_size = 0.3, shuffle = True, random_state=100)
clf3 = SVC(gamma = "auto", C = 1.0,  kernel = "rbf")
clf3.fit(x_train_norm, y_train)

y_pred4_train = clf3.predict(x_train_norm)
y_pred4_test = clf3.predict(x_test_norm)

print('SVM Accuracy: ')
print('SVM - Test: %.2f' % accuracy_score(y_test, y_pred4_test))
print('SVM - Training: %.2f' % accuracy_score(y_train, y_pred4_train))


#################################### Hyperparameter Tuning Q 3.2  #####################################
print('Q 3.2 Hyperparameter Tuning')

clf2 = RandomForestClassifier(random_state=0)
n_estimators_list = np.array([10,20,100])
max_depth_list = np.array([2,4,8])
grid = GridSearchCV(estimator=clf2, cv=10, param_grid=dict(n_estimators=n_estimators_list, max_depth = max_depth_list))
grid.fit(x_train,y_train)


print('Random Forest:')
print('Random Forest - n_estimators: '+ str(n_estimators_list))
print('Random Forest - max_depth: ' + str(max_depth_list))
print('Random Forest - best n_estimator: ' + str(grid.best_estimator_.n_estimators))
print('Random Forest - best max_depth: ' + str(grid.best_estimator_.max_depth))

y_pred_random_grid_test = grid.predict(x_test)

print('Random Forest - Hyperparameter Tuning Accuracy')
print('Random Forest - Before: %.2f' % accuracy_score(y_test, y_pred2_test))
print('Random Forest - After: %.2f' % accuracy_score(y_test, y_pred_random_grid_test))

print('SVM:')

c_list = np.array([0.001, 0.01, 1])
kernel_list = np.array(["rbf","linear"])
clf4 = SVC(gamma="auto")
grid3 = GridSearchCV(estimator = clf4, cv = 10, param_grid = dict(C = c_list, kernel = kernel_list))
grid3.fit(x_train_norm, y_train)

y_pred4_train_grid = grid3.predict(x_train_norm)
y_pred4_test_grid = grid3.predict(x_test_norm)

print('SVM - Kernel Values tested: ' + str(kernel_list))
print('SVM - C values tested: ' + str(c_list))

print('SVM - best kernel: ' + str(grid3.best_estimator_.kernel))
print('SVM - best C: ' + str(grid3.best_estimator_.C))

##################################### Q 3.3 ###################################################################
########################################### Cross-Validation Results #################################################

print('Q 3.3')

print('SVM - Highest mean testing/cv accuracy:' + str(accuracy_score(y_test,y_pred4_test_grid)))
print('SVM - Mean train score: ' + str(np.mean(grid3.cv_results_['mean_test_score'])))
print('SVM - Mean fit time: ' + str(np.mean(grid3.cv_results_['mean_fit_time'])))

##################################### Q 3.4 ###################################################################
#############################  Determine Feature Importance for Random Forest ####################

coefs_list = clf.feature_importances_
col_names = data.columns.values
coef_sorted_list = np.argsort(coefs_list) ### Are these in descending order??
coef_sorted_list2 = []
for i in range(len(data.columns.values)-1):
    coef_sorted_list2.append(col_names[coef_sorted_list[i]])

print('Q 3.4')
print('Random Forest  - Most important feature: ' + str(coef_sorted_list2[0]))
print('Random Forest  - Least important feature: ' + str(coef_sorted_list2[-1]))

##################################### Q 3.5 ###################################################################
#############################  Determine Feature Importance for Random Forest ####################

#Text written in report.txt file.

##################################### Q 3.6 ###################################################################
########################################### Principal Component Analysis #################################################

pca = PCA(n_components=10, svd_solver='full')
pca.fit(x_test)

print('Percent of variance explained: ' + str(pca.explained_variance_))
print('Singular values corresponding to each component: ' + str(pca.singular_values_))