"""module for utility functions for myevaluation, myclassifiers"""
import numpy as np
from tabulate import tabulate

# pylint: disable=consider-using-enumerate
# pylint: disable=invalid-name
def randomize_in_place(alist, parallel_list=None, random_state=None):
    """Randomizes alist and optional parallel_list in place
    Args:
        alist(list of objects): list to randomize
        parallel_list(list of objects): parallel list to randomize
        random_state(int): seed for random generator
    """
    if random_state is not None:
        np.random.seed(random_state)
    for i in range(len(alist)):
        # generate a random index to swap values with
        rand_index = np.random.randint(0, len(alist)) # [0, len(alist))
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] =\
                parallel_list[rand_index], parallel_list[i]

def group_by(X, y):
    """Groups X based on y
    Args:
        X(list of list of objects): data to categorize
        y(list of objects): classes to group based on
    Returns:
        group_names(list of obj): sorted list of no duplicates of classifications
        group_subtables(list of list of obj): table for each category (parallel to group_names)
    """
    group_names = sorted(list(set(y)))
    group_subtables = [[] for _ in group_names] # e.g. [[], [], []]

    for i, row in enumerate(X):
        groupby_val = y[i]
        # which subtable does this row belong?
        groupby_val_subtable_index = group_names.index(groupby_val)
        group_subtables[groupby_val_subtable_index].append(row.copy()) # make a copy

    return group_names, group_subtables

def format_confusion_matrix(matrix, headers=None, index=None):
    """uses tabulate to format a confusion matrix
    Args:
        matrix(list of list of int): confustion matrix to format
    """
    for i, row in enumerate(matrix):
        total = 0
        recog = 0
        for j, col in enumerate(row):
            total += col
            if i == j:
                recog += col
        recog = recog/total *100 if total > 0 else 0
        matrix[i] = row + [total, round(recog,3)]
    new_header = headers + ['Total', 'Recognition (%)']
    print(tabulate(matrix, headers=new_header, showindex=index))

# utility functions for myclassifiers

def simple_high_low_discretizer(val):
    """Return 'high' if val >= 100, 'low' otherwise
    Args:
        val (float): value to be evaluated
    Returns:
        rating (str): string representing a high or low value
    """
    if val >= 100:
        return 'high'
    return 'low'

def compute_euclidean_distance(v_1, v_2):
    """Computes the euclidean distance between two lists of values
    Args:
        v_1 (list[float]): first list of values
        v_2 (list[float]): second list of values
    Returns:
        distances(list[float]): List of floats of distances.
    """
    distance = 0
    for one, two in zip(v_1, v_2):
        try:
            distance += (one - two) ** 2
        except TypeError:
            if one != two:
                distance += 1
    return distance ** 0.5
    # return np.sqrt(sum([(v_1[i] - v_2[i]) ** 2 for i in range(len(v_1))]))

def find_most_frequent_occurence(values):
    """Returns the most frequent item in a list
    Args:
        values (list): list of values to be evaluated
    Returns:
        max_val (obj): most common obj in list
    """
    frequencies = {}
    count, max_val = 0, ''
    for val in values:
        frequencies[val] = frequencies.get(val, 0) + 1
        if frequencies[val] >= count :
            count, max_val = frequencies[val], val
    return max_val

def compute_ratings(table, column):
    """
    Computes the ratings for a column 1 through 10.

    Args:
        table(MyPyTable): MyPyTable containging data
        column(string): the name of the column to compute ratings for
        cutoff(list(int)): a sorted list representing the cutoff points
        for each rating. Each cutoff should represent the maximum value
        allowed to meet the rating (with the exception of the highest rank).
        Values less than or equal to first cutoff will be ranked 1, and values
        greater than or equal to last cutoff will be ranked the highest rank.

    Returns:
        list(int): Parallel list of each value's rating
    """
    cutoffs = [13, 14, 16, 19, 23, 26, 30, 36, 44, 45]
    col_data = table.get_column(column, include_missing_values=False)
    ratings = []
    for val in col_data:
        if val <= cutoffs[0]:
            ratings.append(1)
        elif val >= cutoffs[-1]:
            ratings.append(len(cutoffs))
        else:
            for i in range(1, len(cutoffs)-2):
                if val <= cutoffs[i]:
                    ratings.append(i + 1)
                    break
    return ratings

def discretize_ratings(val):
    """Takes an mpg value and converts it to a classification
    Args:
        val (float): value to be evaluated
    Returns:
        rating (str): string representing an mpg class
    """
    cutoffs = [13, 14, 16, 19, 23, 26, 30, 36, 44, 45]
    length = len(cutoffs)
    rating = -1
    if val <= cutoffs[0]:
        rating = 1
    elif val >= cutoffs[-1]:
        rating = length
    else:
        for i in range(1, length):
            if val <= cutoffs[i]:
                rating = (i + 1)
                break
    return rating

def discretize_ratings_custom(val, cutoffs, output):
    """Takes an mpg value and converts it to a classification
    Args:
        val (float): value to be evaluated
        cutoffs (list of int): cutoffs for the val
        output (list of str): parallel+2 to cutoffs, names for each cutoff
    Returns:
        output (str): string from output
    """
    length = len(cutoffs)
    if val <= cutoffs[0]:
        return output[0]
    elif val >= cutoffs[-1]:
        return output[-1]
    else:
        for i in range(1, length):
            if val <= cutoffs[i]:
                return output[(i + 1)]

def discretize_ratings_normal(val):
    """Takes a normalized mpg value and converts it to a classification
    Args:
        val (float): value to be evaluated
    Returns:
        rating (str): string representing an mpg class
    """
    cutoffs = [13, 14, 16, 19, 23, 26, 30, 36, 44, 45]
    normal_cutoffs = normalize_data(cutoffs)
    length = len(normal_cutoffs)
    rating = -1
    if val <= normal_cutoffs[0]:
        rating = 1
    elif val >= normal_cutoffs[-1]:
        rating = length
    else:
        for i in range(1, length):
            if val <= normal_cutoffs[i]:
                rating = (i + 1)
                break
    return rating

def normalize_data(data):
    """Normalize a row's values
    Args:
        data (list[float]): list of values to normalize
    Returns:
        normalized (list[float]): list of normalized values
    """
    min_value = min(data)
    range_value = max(data) - min(data)
    normalized = []
    for value in data:
        normalized.append((value-min_value)/range_value)
    return normalized

def print_results(title, actual, expected, test_data):
    """Print results for the jupyter notebook
    Args:
        title (str): Title for line 2 of the print
        actual (list[obj]): list of test values
        expected (list[ob]): list of expected values
        test_data (list[obj]): list of test instances
    """
    print(f"""===========================================
{title}
===========================================""")
    num_correct = 0
    num_total = 0
    for row, act, exp in zip(test_data, actual, expected):
        if act == exp:
            num_correct += 1
        num_total += 1
        print(f"""instance: {row}
class: {act} actual: {exp}""")
    print(f"accuracy: {num_correct/num_total}""")


def get_frequencies_2D(input_data, col_index):
    """
    Computes the frequency of each value in a given column

    Args:
        list(list of list): 2D list of data
        col_index(str): index of the column to calculate frequencies on

    Returns:
        values: list of values. Parallel to counts.
        counts: list of counts for each value. Parallel to values.
    """
    col = [row[col_index] for row in input_data]
    col.sort()
    # parallel lists
    values = []
    counts = []
    for value in col:
        if value in values: # seen it before
            counts[-1] += 1 # okay because sorted
        else: # haven't seen it before
            values.append(value)
            counts.append(1)

    return values, counts

def compute_equal_width_cutoffs(values, num_bins):
    """create bins of equal width"""
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins # float
    # since bin_width is a float, we shouldn't use range() to generate a list
    # of cutoffs, use np.arange()
    cutoffs = list(np.arange(min(values), max(values), bin_width))
    cutoffs.append(max(values)) # exactly the max(values)
    # to handle round off error... 
    # if your application allows, we should convert to int
    # or optionally round them
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs

def create_output_for_discrete(X_train, bins=10):
    """create output for the function discretize_ratings_custom"""
    train_cutoffs = compute_equal_width_cutoffs(X_train, bins)
    train_output = [f'X<{train_cutoffs[0]}']
    for i, cut in enumerate(train_cutoffs):
        if i == len(train_cutoffs)-1:
            break
        train_output.append(f'{cut}<=X<{train_cutoffs[i+1]}')
    train_output.append(f'X>{train_cutoffs[-1]}')
    return train_output