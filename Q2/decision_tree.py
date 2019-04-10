from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
        pass

    def learn(self, X, y, n_features):

        def get_split(dataset2, labels2, n_features2, side):
            counter = 0
            ogH = entropy(labels2)
            if ogH != 0.0:
                b_index, b_value, b_score, b_groups = 999, 999, 999, None
                IG_index = 0
                features = list()
                while len(features)< n_features2:
                    index = np.random.random_integers(len(dataset2[0])-2)
                    if index not in features:
                        features.append(index)
                for index in features:
                    colSum = []
                    colMean = np.mean(colSum)
                    for row in dataset2:
                        colSum.append(row[index])
                        X_left, X_right, y_left, y_right = partition_classes(dataset2,labels2,index,row[index])
                        ig = information_gain(labels2,[y_left, y_right])
                        if ig > IG_index:
                            IG_index = ig
                            b_index = index  # row that split occurred on
                            b_value = row[index]  # value that split occurred
                            b_score = ig  # gini index that caused split
                            b_groups = [X_left,y_left,X_right,y_right]
                    if IG_index == 0.0:
                        colMean = np.mean(colSum)
                        X_left, X_right, y_left, y_right = partition_classes(dataset2, labels2, index, colMean)
                        ig = information_gain(labels2, [y_left, y_right])
                        b_index = index  # row that split occurred on
                        b_value = colMean  # value that split occurred
                        b_score = ig  # gini index that caused split
                        b_groups = [X_left, y_left, X_right, y_right]
                return {'index': b_index, 'value': b_value, 'groups': b_groups}
            elif side == 'left':
                b_index = None  # row that split occurred on
                b_value = None  # value that split occurred
                b_score = None  # information gain index that caused split
                b_groups = [dataset2, labels2, None , None]
                return {'index': b_index, 'value': b_value, 'groups': b_groups}
            else:
                b_index = None  # row that split occurred on
                b_value = None  # value that split occurred
                b_score = None  # information gain index that caused split
                b_groups = [None, None, dataset2, labels2]
                return {'index': b_index, 'value': b_value, 'groups': b_groups}

        def split(node, max_depth, min_size, n_features, depth, counter):
            counter = counter + 1

            left = node['groups'][0]
            labels_left = node['groups'][1]
            right = node['groups'][2]
            labels_right = node['groups'][3]


            del (node['groups'])
            # check for a no split
            if not left or not right:
                if not left:
                    node['left'] = node['right'] = to_terminal(right)
                else:
                    node['left'] = node['right'] = to_terminal(left)
                return
            # check for max depth
            if depth >= max_depth:
                node['left'], node['right'] = to_terminal(left), to_terminal(right)
                return
            # process left child
            if len(left) <= min_size:
                node['left'] = to_terminal(left)
            else:
                node['left'] = get_split(left, labels_left, n_features, 'left')
                split(node['left'], max_depth, min_size, n, depth + 1, counter)
            # process right child
            if len(right) <= min_size:
                node['right'] = to_terminal(right)
            else:
                node['right'] = get_split(right, labels_right,n_features, 'right')
                split(node['right'], max_depth, min_size, n_features, depth + 1, counter)

        def to_terminal(group):
            outcomes = [row[-1] for row in group]
            return max(set(outcomes), key=outcomes.count)

        counter = 0
        n = n_features
        max_depth = 10
        min_size = 1
        root = get_split(X,y,n,'both')  # takes the data set and splits

        split(root, max_depth, min_size, n, 1, counter)  #max_depth, min_size, n_features, 1) split doe not return anything just alters root/node
        self.tree = root


    def classify(self, record2):
        def predict(tree_dict, record):
            if tree_dict['value'] == None:
                return int(tree_dict['left'])
            else:
                if record[tree_dict['index']] < tree_dict['value']:
                    if isinstance(tree_dict['left'], dict):
                        return predict(tree_dict['left'], record)
                    else:
                        return int(tree_dict['left'])
                else:
                    if isinstance(tree_dict['right'], dict):
                        return predict(tree_dict['right'], record)
                    else:
                        return int(tree_dict['right'])

        tree_dictionary = self.tree

        return(predict(tree_dictionary, record2))

