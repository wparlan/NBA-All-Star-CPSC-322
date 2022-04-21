"""module containing functions to evaluate classifier accuracy"""
import numpy as np
from mysklearn import myutils
from tabulate import tabulate

# pylint: disable=line-too-long
# pylint: disable=invalid-name
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
    if shuffle:
        myutils.randomize_in_place(X, y, random_state)
    if isinstance(test_size, int):
        if test_size > len(X) or test_size < 0:
            raise ValueError
        split_index = len(X) - test_size
    elif isinstance(test_size, float):
        split_index = len(X) - int(test_size * len(X)) - 1
    else:
        raise ValueError

    X_train = X[0:split_index]
    X_test = X[split_index:]
    y_train = y[0:split_index]
    y_test = y[split_index:]

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
    # set up enviornment
    if random_state is not None:
        np.random.seed(random_state)
    if n_samples is None:
        n_samples = len(X)
    # pick random indices
    choices = list(np.random.choice(range(len(X)), n_samples, replace=True))
    # prepare outputs based on indices
    X_sample, X_out_of_bag = [], []
    if y is not None:
        y_sample, y_out_of_bag = [], []
    else:
        y_sample, y_out_of_bag = None, None
    # assemble outputs based on indices
    for pick in choices:
        X_sample.append(X[pick])
        if y is not None:
            y_sample.append(y[pick])
    # assemble out_of_bag based on indices
    for index in range(len(X)): # pylint: disable=consider-using-enumerate
        if index not in choices:
            X_out_of_bag.append(X[index])
            if y is not None:
                y_out_of_bag.append(y[index])
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

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
    count_true = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            count_true += 1
    if normalize:
        return count_true/len(y_true)
    else:
        return count_true

def accuracy_score_confidence_interval(accuracy, n_samples, confidence_level=0.95):
    """Compute the classification prediction accuracy score confidence interval.

    Args:
        accuracy(float): Classification accuracy to compute a confidence interval for
        n_samples(int): Number of samples in the test set used to compute the accuracy
        confidence_level(float): Level of confidence to use for computing a confidence interval
            0.9, 0.95, and 0.99 are supported. Default is 0.95

    Returns:
        lower_bound(float): Lower bound of the accuracy confidence interval
        upper_bound(float): Upper bound of the accuracy confidence interval

    Notes:
        Raise ValueError on invalid confidence_level
        Assumes accuracy and n_samples are correct based on training/testing
            set generation method used (e.g. holdout, cross validation, bootstrap, etc.)
            See Bramer Chapter 7 for more details
    """
    # check valid confidence level
    if confidence_level not in [0.9, 0.95, 0.99]:
        raise ValueError
    # set z score
    z_score = 0
    if confidence_level == .9:
        z_score = 1.645
    elif confidence_level == .95:
        z_score = 1.960
    else:
        z_score = 2.576
    # calculate the interval
    interval = z_score * ( (accuracy * (1 - accuracy)) / n_samples)**0.5
    return accuracy-interval, accuracy+interval

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
    # parse optional inputs
    if labels is None:
        labels = list(set(sorted(y_true)))
    if pos_label is None:
        pos_label = labels[0]
    # count tp and fp
    num_true_pos, num_false_pos = 0, 0
    for true, pred in zip(y_true, y_pred):
        if true == pos_label and pred == true:
            num_true_pos += 1
        else:
            if pred == pos_label:
                num_false_pos += 1
    return num_true_pos / (num_true_pos+num_false_pos) if (num_true_pos+num_false_pos) > 0 else 0

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
    # parse optional inputs
    if labels is None:
        labels = list(set(sorted(y_true)))
    if pos_label is None:
        pos_label = labels[0]
    # count tp and fn
    num_true_pos, num_false_neg = 0, 0
    for true, pred in zip(y_true, y_pred):
        if true == pos_label:
            if pred == true:
                num_true_pos += 1
            else:
                num_false_neg += 1
    return num_true_pos / (num_true_pos+num_false_neg) if (num_true_pos+num_false_neg) > 0 else 0

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
    # parse optional inputs
    if labels is None:
        labels = list(set(sorted(y_true)))
    if pos_label is None:
        pos_label = labels[0]
    # retrieve precision and recall
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# pylint:disable=unused-argument
def support_score(y_true, y_pred, labels, pos_label):
    """Helper function to calculate support
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
        support(float): support score of the positive class
    """
    support = 0
    for true in y_true:
        if true == pos_label:
            support += 1
    return support

def classification_report(y_true, y_pred, labels=None, output_dict=False):
    """Build a text report and a dictionary showing the main classification metrics.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        output_dict(bool): If True, return output as dict instead of a str

    Returns:
        report(str or dict): Text summary of the precision, recall, F1 score for each class.
            Dictionary returned if output_dict is True. Dictionary has the following structure:
                {'label 1': {'precision':0.5,
                            'recall':1.0,
                            'f1-score':0.67,
                            'support':1},
                'label 2': { ... },
                ...
                }
            The reported averages include macro average (averaging the unweighted mean per label) and
            weighted average (averaging the support-weighted mean per label).
            Micro average (averaging the total true positives, false negatives and false positives)
            multi-class with a subset of classes, because it corresponds to accuracy otherwise
            and would be the same for all metrics.

    Notes:
        Loosely based on sklearn's classification_report():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """
    # parse optional inputs
    if labels is None:
        labels = list(set(sorted(y_true)))
    # set up dictionary
    string_output = []
    output_dictionary = {}
    for label in labels:
        temp_dict = {}
        precision = binary_precision_score(y_true, y_pred, labels, pos_label=label)
        recall = binary_recall_score(y_true, y_pred, labels, pos_label=label)
        f1_score = binary_f1_score(y_true, y_pred, labels, pos_label=label)
        support = support_score(y_true, y_pred, labels, pos_label=label)
        string_output.append([precision, recall, f1_score, support])
        temp_dict['precision'] = precision
        temp_dict['recall'] = recall
        temp_dict['f1_score'] = f1_score
        temp_dict['support'] = support
        output_dictionary[label] = temp_dict
    if output_dict:
        return output_dictionary
    return tabulate(string_output, ['precision', 'recall', 'f1', 'support'], showindex=labels)

def print_classifier_results(classifier_names, classifier_results, test_answers, headers):
    """function to print a bulk list of classifer results with associated names and header
    Args:
        classifier_names(list of str): names for classifiers
        classifier_results(list of list): 1D results for each classifier
        test_answers (list of values): 1D results for each test
        header(list of str): header for confusoin matrices
    """
    for name, result in zip(classifier_names, classifier_results):
        print(f'{name}--------------------------')
        print('Summary:')
        print(f'\tAccuracy..: {round(result[1],3)}')
        print(f'\tError Rate: {round(1-result[1],3)}', '\n')
        print('Precision, Recall, F1:')
        print(classification_report(test_answers, result[0]), '\n')
        matrix = confusion_matrix(test_answers, result[0], headers)
        print('Confusion Matrix:')
        myutils.format_confusion_matrix(matrix, headers=headers, index=headers)
        print('\n')
