from scipy import stats
import numpy as np
import math

def entropy(class_y):

    np_array = class_y
    zero_count = 0
    one_count = 0
    #print(np_array)
    for i in np_array:
        if i == 1:
            one_count = one_count + 1
        elif i == 0:
            zero_count = zero_count + 1
        else:
            print(i & "--Not a 0 or 1.")

    sum_count = zero_count + one_count
    if zero_count == 0:
        p_zero = -0.0
        p_zero_log = 0
    else:
        p_zero = zero_count / sum_count
        p_zero_log = np.log2(p_zero)

    if one_count == 0:
        p_one = -0.0
        p_one_log = 0
    else:
        p_one = one_count / sum_count
        p_one_log = np.log2(p_one)

    entropy = -p_zero * p_zero_log  - p_one * p_one_log
    return entropy

def partition_classes(X, y, split_attribute, split_val):

    list_length = len(X)
    data = X
    labels = y
    X_left = []
    X_right = []
    y_left = []
    y_right = []

    split_type = 'numeric' #either numeric or categorical

    if split_type == 'numeric':
        for index in range(list_length):
            if data[index][split_attribute] <= split_val:
                X_left.append(data[index])
                y_left.append(labels[index])
            else:
                X_right.append(data[index])
                y_right.append(labels[index])
    if split_type == "categorical":
        for index in range(list_length):
            if data[index][split_attribute] == split_val:
                X_left.append(data[index])
                y_left.append(labels[index])
            else:
                X_right.append(data[index])
                y_right.append(labels[index])

    return (X_left, X_right, y_left, y_right)

def information_gain(previous_y, current_y):

    H = entropy(previous_y)
    H_left = entropy(current_y[0])
    P_left = len(current_y[0])/len(previous_y)
    H_Right = entropy(current_y[1])
    P_Right = len(current_y[1])/len(previous_y)

    info_gain = H - (H_left * P_left + H_Right * P_Right)

    return info_gain
