from sympy import false
from mysklearn import myutils
from random import random
from tokenize import group
import numpy as np
import math

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!
    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    X_copy = X
    y_copy = y
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        myutils.randomize_in_place(X_copy, y_copy)
    if type(test_size) == int:
        cutoff = len(X_copy) - test_size
    elif type(test_size) == float:
        cutoff = math.ceil(len(X) * test_size) + 1
    X_train = X_copy[:cutoff]
    X_test = X_copy[cutoff:]
    y_train = y_copy[:cutoff]
    y_test = y_copy[cutoff:]   
    print(X_train, " | ", X_test)
    return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X_train_folds = []
    X_test_folds = []
    split_indices = []
    # create a list of split indices
    n_samples = len(X)
    first_indices = n_samples % n_splits
    current_index = 0
    # from the notes: The first n_samples % n_splits folds have size n_samples // n_splits + 1
    for num in range(first_indices):
        index_pair = [current_index]
        current_index += n_samples // n_splits + 1
        index_pair.append(current_index)
        split_indices.append(index_pair)
    # from the notes: other folds have size n_samples // n_splits, where n_samples is the number of samples
    while len(split_indices) < n_splits:
        index_pair = [current_index]
        current_index += n_samples // n_splits
        index_pair.append(current_index)
        split_indices.append(index_pair)
    # create lists based off splits
    for split in split_indices:
        new_test = list(range(split[0], split[1]))
        new_train = []
        for num in range(len(X)):
            if num not in new_test:
                new_train.append(num)
        # add our 1d arrays to our 2d list
        X_test_folds.append(new_test)
        X_train_folds.append(new_train)
    # shuffle the array and reassign the indices based on their new locations
    if shuffle:
        old_x = list(X)
        myutils.randomize_in_place(X, random_state=random_state)
        # for each index in non-shuffled X, find the equivalent index in the shuffled X
        for i, test in enumerate(X_test_folds):
            for j, index in enumerate(test):
                X_test_folds[i][j] = X.index(old_x[index])
        # again but for train
        for i, train in enumerate(X_train_folds):
            for j, index in enumerate(train):
                X_train_folds[i][j] = X.index(old_x[index])
    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    # built an array of indices
    X_indices = [[i] for i in range(len(X))]
    # set up randomize and shuffle
    if random_state is not None:
        np.random.seed(random_state)
    if shuffle:
        for i, _ in enumerate(X):
            rand_index = np.random.randint(0, len(X))
            X_indices[i], X_indices[rand_index] = X_indices[rand_index], X_indices[i]
    # group by y values
    _, group_subtables = myutils.group_by(X_indices, y)
    # prepare train and test
    X_train_folds, X_test_folds = [], [[] for _ in range(n_splits)]
    split_sizes = []
    first_n_samples = len(X) % n_splits # used for determining test sample size
    # build the sizes of each test set
    for i in range(n_splits):
        if i < first_n_samples:
            split_sizes.append(len(X) // n_splits + 1)
        else:
            split_sizes.append(len(X) // n_splits)
    # create counters to keep track of which index and table
    subtable_id, subtable_index = 0, 0
    # create splits
    for split in range(n_splits):
        # while split is less than the size keep adding on
        while len(X_test_folds[split]) < split_sizes[split]:
            table = group_subtables[subtable_id]
            # catch index out of bounds if one of the subtables is too small
            try:
                X_test_folds[split] += table[subtable_index]
            except IndexError:
                pass
            # cycle through the subtables
            subtable_id = (subtable_id + 1) % len(group_subtables)
            # if you have cycled back to the first table, increment the index
            if subtable_id == 0:
                subtable_index += 1
    # build training sets based on the test sets
    for test_set in X_test_folds:
        new_row = []
        for index in range(len(X)):
            if index not in test_set:
                new_row.append(index)
        X_train_folds.append(new_row)
    # return
    return X_train_folds, X_test_folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.
    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    # if n_samples == None:
    #     n_samples = len(X[0])
    X_sample = []
    X_out_of_bag = []
    y_sample = []
    y_out_of_bag = []
    if random_state is not None:
        np.random.seed(random_state)
    if n_samples == None:
        n_samples = len(X)      
    for i in range(n_samples):
        rand_index = np.random.randint(0, len(X))
        X_sample.append(X[rand_index])
        if y is not None:
            y_sample.append(y[rand_index])
    for i in range(len(X)):
        if X[i] not in X_sample:
            X_out_of_bag.append(X[i])
            if y is not None:
                y_out_of_bag.append(y[i])
    if y is None:
        y_sample = None
        y_out_of_bag = None
        
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag # TODO: fix this

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    for label in labels:
        new_row = [0] * len(labels)
        for true, pred in zip(y_true, y_pred):
            if true == label:
                if pred is None:
                    continue
                new_row[labels.index(pred)] += 1
        matrix.append(new_row)
    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).
    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    num_correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            num_correct += 1
    if normalize:
        return float(num_correct / len(y_pred))
    else:
        return num_correct
    
def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        precision(float): Precision of the positive class
    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    # initialize labels
    if labels == None:
        labels = []
        for val in y_true:
            if val not in labels:
                labels.append(val)
    # initialize pos_label
    if pos_label == None:
        pos_label = labels[0]
    true_positives = 0
    false_positives = 0
    for i in range(len(y_true)):
        if y_pred[i] == pos_label:
            if y_true[i] == y_pred[i]:
                true_positives += 1
            else:
                false_positives += 1
    if true_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0
    return precision
    
def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        recall(float): Recall of the positive class
    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    # initialize labels
    if labels is None:
        labels = []
        for val in y_true:
            if val not in labels:
                labels.append(val)
    # initialize pos_label
    if pos_label == None:
        pos_label = labels[0]
    true_positives = 0
    false_negatives = 0
    for i in range(len(y_true)):
        if y_pred[i] == pos_label:
            if y_true[i] == y_pred[i]:
                true_positives += 1
        else:
            if y_true[i] == pos_label:
                false_negatives += 1
    if true_positives > 0:
        precision = true_positives / (true_positives + false_negatives)
    else:
        precision = 0
    return precision
    

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        f1(float): F1 score of the positive class
    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    if precision + recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
