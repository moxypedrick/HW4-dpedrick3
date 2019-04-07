from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
        pass

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)

        #I should pass get_split a x and y. This x and y

        def get_split(dataset, labels):
            counter = 0
            #class_values = list(set(row[-1] for row in dataset))
            b_index, b_value, b_score, b_groups = 999, 999, 999, None
            IG_index = 0
            features = list()
            n_features = int(np.sqrt(len(dataset[0])-1))
            while len(features)< n_features:
                index = np.random.random_integers(len(dataset[0])-1)
                if index not in features:
                    features.append(index)
            #print(len(features))
            #print(features)
            #print(y[0])
            for index in features:
                for row in dataset:
                    #print(row[index],index)
                    X_left, X_right, y_left, y_right = partition_classes(dataset,labels,index,row[index])
                    #print(y_left)
                    #print(y_right)
                    ig = information_gain(labels,[y_left, y_right])
                    #print(IG_index,ig)

                    if ig > IG_index:
                        b_index = index  # row that split occurred on
                        b_value = row[index]  # value that split occurred
                        b_score = ig  # gini index that caused split
                        b_groups = [X_left,y_left,X_right,y_right]  # the split group
            #print({'index': b_index, 'value': b_value, 'groups': b_groups})
            return {'index': b_index, 'value': b_value, 'groups': b_groups}

        def split(node, max_depth, min_size, n_features, depth, counter):
            counter = counter + 1
            print(counter, max_depth, min_size,n_features,depth)
            left,labels_left,right,labels_right = node['groups']

            del (node['groups'])
            # check for a no split
            if not left or not right:
                node['left'] = node['right'] = to_terminal(left + right)
                return
            # check for max depth
            if depth >= max_depth:
                node['left'], node['right'] = to_terminal(left), to_terminal(right)
                return
            # process left child
            if len(left) <= min_size:
                node['left'] = to_terminal(left)
            else:
                node['left'] = get_split(left, labels_left)
                split(node['left'], 2, min_size, 2, depth + 1, counter)
            # process right child
            if len(right) <= min_size:
                node['right'] = to_terminal(right)
            else:
                node['right'] = get_split(right, labels_right)
                split(node['right'], 2, min_size, 2, depth + 1, counter)

        def to_terminal(group):
            outcomes = [row[-1] for row in group]
            return max(set(outcomes), key=outcomes.count)

        counter = 0
        print(counter)
        root = get_split(X,y)  # takes the data set and splits
        print('finished root')

        split(root, 2, 2, len(y), 1, counter)  #max_depth, min_size, n_features, 1) split doe not return anything just alters root/node
        return root




    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        def predict(node, row):
            if row[node['index']] < node['value']:
                if isinstance(node['left'], dict):
                    return predict(node['left'], row)
                else:
                    return node['left']
            else:
                if isinstance(node['right'], dict):
                    return predict(node['right'], row)
                else:
                    return node['right']

        def bagging_predict(trees, row):
            predictions = [predict(tree, row) for tree in trees]
            return max(set(predictions), key=predictions.count)

        pass
