from decision_tree import DecisionTree
import csv
import numpy as np  # http://www.numpy.org
import ast

class RandomForest(object):
    num_trees = 0
    n_features = 0

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
            #print(i)
            samples.append(XX[i])#[:-1])
            labels.append(XX[i][-1])

        return (samples, labels)

    def bootstrapping(self, XX):
        self.n_features = int(np.sqrt(len(XX[0]) - 1))
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)

    def fitting(self):
        for i in range(self.num_trees):
            tree = self.decision_trees[i]
            tree.learn(self.bootstraps_datasets[i], self.bootstraps_labels[i], self.n_features)
        pass

    def voting(self, X):
        y = []
        count = 0
        for record in X:
            votes = []
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]
                dataset2 = []
                for p in dataset:
                    dataset2.append(p[:-1])  # [:-1])
                if record not in dataset2:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)
            counts = np.bincount(votes)

            if len(counts) == 0:
                not_OOB_tree = self.decision_trees[-1]
                effective_vote_2 = not_OOB_tree.classify(record)
                y = np.append(y, effective_vote_2)
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

    forest_size = 15
    # Initializing a random forest.
    randomForest = RandomForest(forest_size)


    # Creating the bootstrapping datasets
    print("creating the bootstrap datasets")
    randomForest.bootstrapping(XX)

    # Building trees in the forest
    print("fitting the forest")
    randomForest.fitting()

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
