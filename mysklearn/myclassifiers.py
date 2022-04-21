"""
Collection of simple classifiers
"""
from cmath import log
from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

# pylint:disable=line-too-long
# pylint:disable=invalid-name

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, x_train, y_train):
        """Fits a simple linear regression line to x_train and y_train.

        Args:
            x_train(list of list of numeric vals): The list of training instances (samples).
                The shape of x_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to x_train)
                The shape of y_train is n_train_samples
        """
        if self.regressor is None:
            x_train = [x[0] for x in x_train] # convert 2D list with 1 col to 1D list
            slope, intercept = MySimpleLinearRegressor.compute_slope_intercept(x_train,
                y_train)
            self.regressor = MySimpleLinearRegressor(slope, intercept)
        else:
            self.regressor.fit(x_train, y_train)

    def predict(self, x_test):
        """Makes predictions for test samples in x_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            x_test(list of list of numeric vals): The list of testing samples
                The shape of x_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to x_test)
        """
        if not self.regressor:
            return []

        discrete_predictions = []
        predictions = self.regressor.predict(x_test)
        for prediction in predictions:
            discrete_predictions.append(self.discretizer(prediction))
        return discrete_predictions

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        x_train(list of list of numeric vals): The list of training instances (samples).
                The shape of x_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to x_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        """Fits a kNN classifier to x_train and y_train.

        Args:
            x_train(list of list of numeric vals): The list of training instances (samples).
                The shape of x_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to x_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores x_train and y_train
        """
        self.x_train = x_train
        self.y_train = y_train

    def kneighbors(self, x_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            x_test(list of list of numeric vals): The list of testing samples
                The shape of x_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in x_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in x_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        # go through every test value
        for test_value in x_test:
            distance_dict = {}
            # calculate the distances of test_val to x_train
            for index, train_value in enumerate(self.x_train):
                distance_dict[index] = myutils.compute_euclidean_distance(train_value, test_value)
            # sort for only the n_neighbor closest
            nearest_distance = []
            nearest_index = []
            count = 0
            # sort the dict keys (distances), collect the first n_neighbors
            sorted_dict = {k: v for k, v in sorted(distance_dict.items(), key=lambda item: item[1])}
            for key in sorted_dict:
                if count >= self.n_neighbors:
                    break
                nearest_distance.append(sorted_dict[key])
                nearest_index.append(key)
                count += 1
            # append the closest to the returned lists
            distances.append(nearest_distance)
            neighbor_indices.append(nearest_index)
        return distances, neighbor_indices

    def predict(self, x_test):
        """Makes predictions for test instances in x_test.

        Args:
            x_test(list of list of numeric vals): The list of testing samples
                The shape of x_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to x_test)
        """
        y_predicted = []
        _, neighbor_indices = self.kneighbors(x_test)
        for index in neighbor_indices:
            values = []
            for i in index:
                values.append(self.y_train[i])
            max_val = myutils.find_most_frequent_occurence(values)
            y_predicted.append(max_val)
        return y_predicted


class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        x_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of x_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, _, y_train):
        """Fits a dummy classifier to x_train and y_train.

        Args:
            x_train(list of list of numeric vals): The list of training instances (samples).
                The shape of x_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to x_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        # x does not matter, just need most common label from y
        self.most_common_label = myutils.find_most_frequent_occurence(y_train)

    def predict(self, x_test):
        """Makes predictions for test instances in x_test.

        Args:
            x_test(list of list of numeric vals): The list of testing samples
                The shape of x_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to x_test)
        """
        return [self.most_common_label] * len(x_test)

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # for priors (y_train): make a dictionary
        # for posteriors (X_train): make multi dimensional dictionary -- priors:attr index:attr value:attr freq
        priors = {}
        posteriors = {}
        group_names, group_subtables = myutils.group_by(X_train, y_train)
        total = len(X_train)
        # builds priors
        for value in sorted(y_train):
            priors[value] = priors.get(value, 0) + 1
        for key, value in priors.items():
            priors[key] = value / len(y_train)
            posteriors[key] = {}
        # builds posteriors using group_by
        for index, value in enumerate(group_names):
            subtable = group_subtables[index] # parallel
            # for each x instance in subtable, compute frequencies of each column
            for col_index in range(len(subtable[0])):
                internal_dictionary = {}
                names, frequencies = myutils.get_frequencies_2D(subtable, col_index)
                total = 0
                for val in frequencies:
                    total += val
                for i, name in enumerate(names):
                    internal_dictionary[name] = frequencies[i]/total
                posteriors[value][col_index] = internal_dictionary
        self.priors = priors
        self.posteriors = posteriors

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # create return value and a list to map classifications to indices
        y_predicted = []
        classifications = list(self.priors)
        # go through x test and find largest probability from priors*posteriors
        for test_val in X_test:
            probabilities = []
            # create a temp array of each different priors' value
            for prior_key, prior_value in self.priors.items():
                new_prob = prior_value
                # calculate prob by multiplying posterior values
                for index, val in enumerate(test_val):
                    try:
                        new_prob *= (self.posteriors[prior_key][index][val])
                    except KeyError:
                        new_prob = 0
                probabilities.append(new_prob)
            # append the max probabilities' classification to y_pre
            # check if there is > 1 max
            new_classification = ""
            if probabilities.count(max(probabilities)) > 1:
                # pick the prior with the highest chance if there's a tie
                max_indices = []
                highest_chance = 0
                prior_chances = list(self.priors.values())
                for index, val in enumerate(probabilities):
                    if val == max(probabilities):
                        max_indices.append(index)
                for index in max_indices:
                    if prior_chances[index] > highest_chance:
                        highest_chance = prior_chances[index]
                        new_classification = classifications[index]
            else:
                new_classification = classifications[probabilities.index(max(probabilities))]
            y_predicted.append(new_classification)
        return y_predicted

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

    def all_same_class(self, att_partition):
        """checks to see if a partitoin contains all of the same class label
        Args:
            att_partition (list of list of obj): list of X_train instances
        Returns:
            all_same (bool): true if all instances have same classification
        """
        init_val = att_partition[0][-1]
        for instance in att_partition:
            if instance[-1] != init_val:
                return False
        return True

    def find_priors_and_posteriors(self, instances, attribute_domain, attribute_index):
        """calculates the odds of selecting a given attribute value within a given index
        Args:
            instances(list of list): training instances
            attribute_domain(dict int:list): dictionary mapping index to possible values
            attribute_index(int): index to calculate odds for
        Returns:
            priors(dict obj:float): odds of picking a given attr
            posteriors(dict obj: obj:float): odds of picking a given value and class
        """
        # set up for group by
        domain = sorted(attribute_domain[attribute_index])
        classifications = list(set(sorted([row[-1] for row in instances])))
        group_by_classifications = [row[attribute_index] for row in instances]
        # perform group by
        _, group_subtables = myutils.group_by(instances, group_by_classifications)
        priors = {}
        posteriors = {}
        # go through domain and set up priors
        for i, d in enumerate(domain):
            try:
                priors[d] = len(group_subtables[i])/len(instances)
            except IndexError:
                priors[d] = 0
            posteriors[d] = {}
        # go through domain and set up posteriors
        for i, d in enumerate(domain):
            # go through the different classifications
            for c in classifications:
                try:
                    total = len(group_subtables[i])
                except IndexError:
                    continue
                else:
                    # calculate the odds
                    num_pos = 0
                    for instance in group_subtables[i]:
                        if instance[-1] == c:
                            num_pos+=1
                    posteriors[d][c] = num_pos/total
        # return
        return priors, posteriors

    def select_attribute(self, attribute_domains, instances, attributes):
        """select the attribute with highest entropy
        Args:
            instances (list of list of obj): list of X_train instances
            attributes (list of obj): list of availible attributes to select
        Returns:
            selected_attribute (obj): attribute with highest entropy
        """
        # set up
        e_new = []
        classifications = sorted(list(set([row[-1] for row in instances])))
        attribute_indices = [int(attr[-1]) for attr in attributes]
        # for each attribute's possible values, calculate its entropy
        for index in attribute_indices:
            # calculate priors and posteriors for this column
            domain = sorted(attribute_domains[index])
            priors, posteriors = self.find_priors_and_posteriors(instances, attribute_domains, index)
            entropy = []
            # go through priors/posteriors data to calculate attribute's entropy
            for d in domain:
                # entropy of a given attribute value
                e = 0
                for c in classifications:
                    try:
                        new_val = posteriors[d][c]
                        # if a posterior is 0, the entropy is 0
                        if new_val == 0:
                            e = 0
                            break
                        new_e = -new_val*log(new_val,2).real
                        e += new_e
                    except KeyError:
                        e = 0
                        break
                entropy.append(e)
            # calculate e_new based on the entropy for each of the attribute's values
            e_final = 0
            prior_list = [priors[i] for i in sorted(list(priors))]
            for prior, entropy_val in zip(prior_list, entropy):
                e_final += prior*entropy_val
            e_new.append(e_final)
        # return the attribute with the index of the lowest e_new
        selected_attribute = attributes[e_new.index(min(e_new))]
        return selected_attribute

    def partition_instances(self, header, attribute_domains, instances, split_attribute):
        """group instances based on split_attribute
        Args:
            header (list of str): header row of the data
            attribute_domains (dict str:list of str): possible values for each attribute
            instances (list of list of obj): list of X_train instances
            split_attribute (obj): attribute to split instances on
        Returns:
            partitions (dict str:list of obj): dictionary mapping attribute values to instances
        """
        # this is a group by attribute domain
        # let's use a dictionary
        partitions = {} # key (attribute value): value (subtable)
        att_index = header.index(split_attribute) # e.g. level -> 0
        att_domain = sorted(attribute_domains[att_index]) # e.g. ["Junior", "Mid", "Senior"]
        for att_value in att_domain:
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions

    def tdidt(self, header, attribute_domains, current_instances, available_attributes):
        """Top Down Inductive of Decision Trees algorithm. Recursive func to create a tree
        Args:
            header (list of str): header row of the data
            attribute_domains (dict str:list of str): possible values for each attribute
            current_instances (list of list of obj): list of X_train instances
            availible_attributes (obj): attribute to split instances on
        Returns:
            tree (list): the decision tree
        """
        # select an attribute to split on
        attribute = self.select_attribute(attribute_domains, current_instances, available_attributes)
        available_attributes.remove(attribute) # can't split on this again in
        # this subtree
        tree = ["Attribute", attribute] # start to build the tree!!

        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(header, attribute_domains, current_instances, attribute)
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value, att_partition in partitions.items():
            value_subtree = ["Value", att_value]
            # CASE 1: all class labels of the partition are the same => make a leaf node
            if len(att_partition) > 0 and self.all_same_class(att_partition):
                # make a leaf node
                # count number of instances in all partitions
                num_values = 0
                for partition in partitions.values():
                    num_values += len(partition)
                # create leaf and append
                leaf = ['Leaf', att_partition[0][-1], len(att_partition), num_values]
                value_subtree.append(leaf)
            # CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                # get frequencies of the class attributes from the partition
                class_values, class_counts = myutils.get_frequencies_2D(att_partition, -1)
                # select the attribute with the highest (alphabetical if tie)
                majority_vote = class_values[class_counts.index(max(class_counts))]
                # count total
                num_values = 0 # I THINK LEN(CURRENT_INSTANCES) IS THE SAME
                for partition in partitions.values():
                    num_values += len(partition)
                # create leaf and append
                leaf = ['Leaf', majority_vote, len(att_partition), len(current_instances)]
                value_subtree.append(leaf)
            # CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                # I think this means replace tree with a leaf node that is majority of current_instances
                # get frequencies of the class attributes from the partition
                class_values, class_counts = myutils.get_frequencies_2D(current_instances, -1)
                # select the attribute with the highest (alphabetical if tie)
                majority_vote = class_values[class_counts.index(max(class_counts))]
                # count total
                num_values = 0
                for partition in partitions.values():
                    num_values += len(partition)
                # replace tree with leaf
                tree = ['Leaf', majority_vote, len(current_instances), 0]
                return tree
            else: # none of the previous conditions were true... recurse!
                subtree = self.tdidt(header, attribute_domains, att_partition, available_attributes.copy())
                if subtree[0] == 'Leaf':
                    # check to make sure there are no other leafs with same value
                    # subtree[0] = 'LEAF CASE 3'
                    subtree[-1] = len(current_instances)
                # note the copy
                value_subtree.append(subtree)
            tree.append(value_subtree)
        return tree

    def fit(self, X_train, y_train):
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
        # save x_train and y_train
        self.X_train = X_train
        self.y_train = y_train
        # create a header programatically
        header = [f'attr{i}' for i in range(len(X_train[0]))]
        # create an attribute domain (index:[possible values])
        attribute_domains = {}
        for index in range(len(X_train[0])):
            column_values = []
            for row in X_train:
                column_values.append(row[index])
            attribute_domains[index] = list(set(sorted(column_values)))
        # next, I advise stitching X_train and y_train together
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # now, making a copy of the header because tdidt()
        # is going to modify the list
        available_attributes = header.copy()
        # recall: python is pass by object reference
        tree = self.tdidt(header, attribute_domains, train, available_attributes)
        # note: the unit test will assert tree == interview_tree_solution
        # (mind the attribute value order)
        self.tree = tree

    def tdidt_predict(self, tree, instance):
        """recursive helper function to predict
        Args:
            tree (list of list): current working subtree
            instance (list): instance to predict
        Returns:
            prediction(obj): prediction for the instance
        """
        # go through tree (on attribute node)
        # identify which attribute to split on ('attr#')
        # call again until we reach a root
        # return the prediction from the root
        if tree[0] == 'Leaf':
            return tree[1]
        split_attribute_index = int(tree[1][-1]) # last char from attr# string
        split_attribute = instance[split_attribute_index]
        new_tree = []
        for working_tree in tree:
            try:
                if working_tree[0] == 'Value' and working_tree[1] == split_attribute:
                    new_tree = working_tree[2]
                    break
                elif working_tree[0] == 'Leaf' and working_tree[1] == split_attribute:
                    return working_tree[1]
            except TypeError:
                continue
        return self.tdidt_predict(new_tree, instance)


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for instance in X_test:
            prediction = self.tdidt_predict(self.tree, instance)
            predictions.append(prediction)
        return predictions

    def recursive_print(self, tree, attributes, values, attribute_names, class_name):
        """recursive helper function to parse decision tree rules
        Args:
            tree(nested list): current working subtree
            attributes(list of str): collection of """
        # go through indices of tree
        # build parallel lists of attributes and values
        # where a rule is "if a[i] == v[i] AND a[i+1] == v[i+1] ...""
        valid_start = ['Attribute', 'Value', 'Leaf']
        if tree[0] not in valid_start:
            return
        for index, value in enumerate(tree):
            try:
                if value == 'Attribute':
                    name_index = int(tree[index+1][-1])
                    attributes.append(attribute_names[name_index])
                    for i in range(index, len(tree)):
                        self.recursive_print(tree[index+i], attributes.copy(), values.copy(), attribute_names, class_name)
                elif value == 'Value':
                    values.append(tree[index+1])
                    self.recursive_print(tree[index+2], attributes.copy(), values.copy(), attribute_names, class_name)
                elif value == 'Leaf':
                    rule = "IF "
                    for attr, val in zip(attributes, values):
                        rule += f"{attr} == {val} AND "
                    rule = rule[:-4] + "THEN "
                    rule += f'{class_name} == {tree[index+1]}'
                    print(rule)
                    return
            except TypeError:
                continue

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
            attribute_names = [f'attr{i}' for i in range(len(self.X_train[0]))]
        attributes, values = [], []
        self.recursive_print(self.tree, attributes, values, attribute_names, class_name)
        return

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        # BONUS
        pass # pylint:disable=unnecessary-pass
