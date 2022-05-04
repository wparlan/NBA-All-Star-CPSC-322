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

    n_samples = len(X)
    X_copy = list(X)
    switch_form = n_samples % n_splits
    print(n_samples, n_splits, switch_form)
    start_index = 0
    end_index = 0    
    X_train_folds = []
    X_test_folds = []
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        myutils.randomize_in_place(X_copy)   
    for i in range(n_splits):
        test_list = []
        train_list = []
        if i < switch_form:
            end_index = n_samples // n_splits + 1 + start_index
        else:
            end_index = n_samples // n_splits + start_index
            
        if end_index == n_samples - 1 and i == n_splits - 1:
            test_list = [x for x in range(start_index, n_samples)]
        else:
            test_list = [x for x in range(start_index, end_index)]   
        start_index = end_index
        for j in range(len(X_copy)):
            if j not in test_list:
                train_list.append(j)
        if shuffle:
            for j in range(len(train_list)):
                train_list[j] = X_copy.index(X[train_list[j]])
            for j in range(len(test_list)):
                test_list[j] = X_copy.index(X[test_list[j]])              
        X_train_folds.append(train_list)
        print(i, "| Train Fold:", train_list)
        X_test_folds.append(test_list)
        print(i, "| Test Fold:", test_list)
        
    print( "Train Folds:", X_train_folds)
    print( "Test Folds:", X_test_folds)
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
    n_samples = len(X)
    X_copy = list(X)
    y_copy = list(y)
    print(n_samples, n_splits)
    print(X, " | ", y_copy)
    start_index = 0
    end_index = 0    
    X_train_folds = []
    X_test_folds = []
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        myutils.randomize_in_place(X_copy, y_copy)   
    group_names, group_subtables = myutils.group_by(X_copy, y_copy)
    print("Group names:", group_names, " |  Group subtables:", group_subtables)
    curr_classifier_indexes = [0 for x in group_names]
    curr_classfier = 0
    for i in range(n_splits):
        if i < n_samples % n_splits:
            size = n_samples // n_splits + 1
        else:
            size = n_samples // n_splits
        
        train_list = []
        test_list = []
        for j in range(size):
            if curr_classifier_indexes[curr_classfier] < len(group_subtables[curr_classfier]):
                test_list.append(X_copy.index(group_subtables[curr_classfier][curr_classifier_indexes[curr_classfier]]))
                curr_classifier_indexes[curr_classfier] += 1
            else:
                size += 1
                
            curr_classfier += 1
            if curr_classfier == len(group_names):
                curr_classfier = 0
                
        train_list = []
        for j in range(len(X_copy)):
            if j not in test_list:
                train_list.append(j)
        
        if shuffle:
            for j in range(len(train_list)):
                train_list[j] = X_copy.index(X[train_list[j]])
            for j in range(len(test_list)):
                test_list[j] = X_copy.index(X[test_list[j]])
        
        print(i, "| Train Fold:", train_list)       
        X_train_folds.append(train_list)
        X_test_folds.append(test_list)
        print(i, "| Test Fold:", test_list)
    print( "Train Folds:", X_train_folds)
    print( "Test Folds:", X_test_folds)
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
    print(y_true)
    print(y_pred)
    print(labels)
    matrix = []
    for i in range(len(labels)):
        row = [0 for x in labels]
        for j in range(len(y_true)):
            if(y_true[j] == labels[i]):
                row[labels.index(y_pred[j])] += 1
        matrix.append(row)
        print(i, " |", row)      
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
