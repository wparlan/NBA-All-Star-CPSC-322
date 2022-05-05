import operator
from bitarray import test
import math

import py
from mysklearn import myutils
from mysklearn import myevaluation
class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train, F):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        attributes = []
        for i in range(len(X_train[0])):
            attribute = "att" + str(i)
            attributes.append(attribute) 
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = attributes.copy()
        self.tree = myutils.tdidt(train, train, available_attributes, len(train), F)
        

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        majority = 0
        maj_value = 0
        at_leaf = ""
        traverse_tree = self.tree.copy()
        while at_leaf != "Leaf":
                for i in range(2, len(traverse_tree)):
                    if traverse_tree[0] == "Leaf":
                        at_leaf = "Leaf"
                        if traverse_tree[2] > maj_value:
                            maj_value = traverse_tree[2]
                            majority = traverse_tree[1]
                        break
                    else:
                        traverse_tree = traverse_tree[2][2]
        for instance in X_test:
            at_leaf = ""
            head_attribute = self.tree[1]
            print("Head Attribute", head_attribute)
            index = int(head_attribute[-1])
            traverse_tree = self.tree.copy()
            while at_leaf != "Leaf":
                for i in range(2, len(traverse_tree)):
                    if traverse_tree[i][1] == instance[index]:
                        traverse_tree = traverse_tree[i][2]
                        if traverse_tree[0] == "Attribute":
                            index = int(traverse_tree[1][-1])
                        if traverse_tree[0] == "Leaf":
                            at_leaf = "Leaf"
                            y_predicted.append(traverse_tree[1])
                        break
                    elif i == len(traverse_tree) - 1:
                        print("BREAKING POINT")
                        y_predicted.append(majority)
                        at_leaf = "Leaf"

        print("y_predicted:", y_predicted)
        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.
        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names is None:
            attributes = []
            for i in range(len(self.X_train[0])):
                attribute = "att" + str(i)
                attributes.append(attribute)
            attribute_names = attributes
        attributes, values = [], []
        myutils.find_rules_recursive(self.tree, attributes, values, attribute_names, class_name)

class MyRandomForestClassifier:
    """Represents a random forest classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        trees(nested list): List of decision trees in forest
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyRandomForestClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.trees = []
        self.N = None
        self.M = None
        self.F = None

    def fit(self, X_train, y_train, N, F, M):
        """Fits a random forest classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # assign class variables to parameters
        self.X_train = X_train
        self.y_train = y_train
        self.N = N
        self.M = M
        self.F = F
        
        train_folds, test_folds = myevaluation.stratified_kfold_cross_validation(self.X_train, self.y_train, 3, 0, True)
        train_fold = train_folds[0]
        test_fold = test_folds[0]
        # print("Train:", train_fold)
        # print("Test:", test_fold)
        X_test_set = []
        y_test_set = []
        for index in test_fold:
            X_test_set.append(X_train[index])
            y_test_set.append(y_train[index])
        X_remainder_set = []
        y_remainder_set = []
        for index in train_fold:
            X_remainder_set.append(X_train[index])
            y_remainder_set.append(y_train[index])
        # print("Y REMAINDER", y_remainder_set, len(y_remainder_set))
        # bootstrap samples
        n_samples = math.ceil(len(X_train) * 63 / 100)
        X_train_sets = []
        X_valid_sets = []
        y_train_sets = []
        y_valid_sets = []
        for i in range(N):
            X_training_set, X_validation_set, y_training_set, y_validation_set = myevaluation.bootstrap_sample(X_remainder_set, y_remainder_set, n_samples, i)
            X_train_sets.append(X_training_set)
            X_valid_sets.append(X_validation_set)
            y_train_sets.append(y_training_set)
            y_valid_sets.append(y_validation_set)
        # print("XTRAINSET", X_train_sets)
        # print("XVALIDSET:", X_valid_sets)
        
        decision_tree = MyDecisionTreeClassifier()   
        y_predicteds = []
        for i in range(len(X_train_sets)):
            decision_tree.fit(X_train_sets[i], y_train_sets[i], F)
            self.trees.append(decision_tree.tree)
            # print(decision_tree.tree)
            # print(X_valid_sets[i])
            y_predicted = decision_tree.predict(X_valid_sets[i])
            # print(y_valid_sets[i])
            y_predicteds.append(y_predicted)
        accuracy_scores = []
        for i in range(len(y_predicteds)):
            score = myevaluation.accuracy_score(y_valid_sets[i], y_predicteds[i])
            accuracy_scores.append(score)
        # print(accuracy_scores)
        indexes = myutils.Nmaxelements(accuracy_scores.copy(), M)
        # print(indexes)
        new_trees = []
        for index in indexes:
            new_trees.append(self.trees[index])
        self.trees = new_trees    
        print(new_trees)
             
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_total_predicted = []
        counts = {}
        for i in range(len(self.trees)):
            decision_tree = MyDecisionTreeClassifier()
            decision_tree.tree = self.trees[i]
            y_total_predicted.append(decision_tree.predict(X_test))
        for i in range(len(X_test)):
            title = "num" + str(i)
            counts[title] = {}
            for j in range(len(y_total_predicted)):
                value = y_total_predicted[j][i]
                if value in counts[title]:
                    counts[title][value] += 1
                else:
                    counts[title][value] = 1
        y_predicted = []
        for item in counts:
            max = 0
            max_value = 0
            dict = counts[item]
            for value in dict:
                if dict[value] > max:
                    max_value = value
                    max = dict[value]
            y_predicted.append(max_value)    
        # print(counts)
        # print(y_predicted)
        return(y_predicted)
