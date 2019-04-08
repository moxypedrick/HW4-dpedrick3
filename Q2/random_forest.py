from decision_tree import DecisionTree
import csv
import numpy as np  # http://www.numpy.org
import ast

class RandomForest(object):
    num_trees = 10
    decision_trees = []
    bootstraps_datasets = []
    bootstraps_labels = []

    def __init__(self, num_trees):
        # Initialization done here
        self.num_trees = num_trees
        self.decision_trees = [DecisionTree() for i in range(num_trees)]

    def _bootstrapping(self, XX, n):
        values = np.random.randint(0, len(XX), n)
        samples = []  # sampled dataset
        labels = []  # class labels for the sampled records

        for i in values:
            samples.append(XX[i])
            labels.append(XX[i][-1])

        return (samples, labels)


    def bootstrapping(self, XX):
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)


    def fitting(self, n_features):
        for i in range(self.num_trees):
            tree = self.decision_trees[i]
            tree.learn(self.bootstraps_datasets[i], self.bootstraps_labels[i], n_features)
        pass

    def voting(self, X):
        y = []
        for record in X:
            # Following steps have been performed here:
            #   1. Find the set of trees that consider the record as an 
            #      out-of-bag sample.
            #   2. Predict the label using each of the above found trees.
            #   3. Use majority vote to find the final label for this record.
            votes = []
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]

                if record not in dataset:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)
            counts = np.bincount(votes)
            
            if len(counts) == 0:
                print('here')
                # TODO: Special case 
                #  Handle the case where the record is not an out-of-bag sample
                #  for any of the trees. 
                pass
            else:
                y = np.append(y, np.argmax(counts))

        return y

# DO NOT change the main function apart from the forest_size parameter!
def main():
    X = list()
    y = list()
    XX = list()
    numerical_cols = numerical_cols=set([i for i in range(0,43)]) # indices of numeric attributes (columns)

    # Loading data set
    print("reading hw4-data")
    with open("hw4-data.csv") as f:
        for line in csv.reader(f, delimiter=","):
            xline = []
            for i in range(len(line)):
                if i in numerical_cols:
                    xline.append(ast.literal_eval(line[i]))
                else:
                    xline.append(line[i])

            X.append(xline[:-1])
            y.append(xline[-1])
            XX.append(xline[:])

    # TODO: Initialize according to your implementation
    # VERY IMPORTANT: Minimum forest_size should be 10
    forest_size = 15
    num_trees = 10
    decision_trees = []
    n_features = int(np.sqrt(len(XX[0])-1))

    # the bootstrapping datasets for trees
    # bootstraps_datasets is a list of lists, where each list in bootstraps_datasets is a bootstrapped dataset.
    bootstraps_datasets = []

    # the true class labels, corresponding to records in the bootstrapping datasets
    # bootstraps_labels is a list of lists, where the 'i'th list contains the labels corresponding to records in
    # the 'i'th bootstrapped dataset.
    bootstraps_labels = []

    # Initializing a random forest.
    randomForest = RandomForest(forest_size)

    # Creating the bootstrapping datasets
    print("creating the bootstrap datasets")
    randomForest.bootstrapping(XX)


    # Building trees in the forest
    print("fitting the forest")
    randomForest.fitting(n_features)

    # Calculating an unbiased error estimation of the random forest
    # based on out-of-bag (OOB) error estimate.
    y_predicted = randomForest.voting(X)

    # Comparing predicted and true labels
    results = [prediction == truth for prediction, truth in zip(y_predicted, y)]

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))

    print("accuracy: %.4f" % accuracy)
    print("OOB estimate: %.4f" % (1-accuracy))


if __name__ == "__main__":
    main()
