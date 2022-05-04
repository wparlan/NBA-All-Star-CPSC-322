import numpy as np
from math import log2
from attr import attrib
import numpy as np
from sqlalchemy import false, true
import random


def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
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

def compute_euclidean_distance(v1, v2):
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

def compute_slope_intercept(x, y):
    meanx = np.mean(x)
    meany = np.mean(y)

    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    den = sum([(x[i] - meanx) ** 2 for i in range(len(x))])
    m = num / den 
    # y = mx + b => b = y - mx 
    b = meany - m * meanx
    return m, b

def normalize(X_train, X_test):
    min1 = X_train[0][0]
    max1 = X_train[0][0]
    min2 = X_train[0][1]
    max2 = X_train[0][1]
    
    for row in X_train:
        if row[0] > max1:
            max1 = row[0]
        if row[0] < min1:
            min1 = row[0]
        if row[1] > max2:
            max2 = row[1]
        if row[1] < min2:
            min2 = row[1]
    
    range1 = max1 - min1
    range2 = max2 - min2
    
    for row in X_train:
        row[0] = (row[0] - min1) / range1
        row[1] = (row[1] - min1) / range2
    
    X_test[0][0] = (X_test[0][0] - min1) / range1
    X_test[0][1] = (X_test[0][1] - min2) / range2
    
    return X_train, X_test

def discretizer(vals):
    y_discretized = []
    for val in vals:
        if val >= 100:
            y_discretized.append("high")
        else:
            y_discretized.append("low")
    return y_discretized

def get_mpg_rating(vals):
    for row in range(len(vals)):
        if vals[row] <= 14.0:
            return 1
        elif vals[row] == 14.0:
            return 2  
        elif vals[row] > 14.0 and vals[row] <= 16.0:
            return 3 
        elif vals[row] > 16.0 and vals[row] <= 19.0:
            return 4 
        elif vals[row] > 19.0 and vals[row] <= 23.0:
            return 5 
        elif vals[row] > 23.0 and vals[row] <= 26.0:
            return 6 
        elif vals[row] > 26.0 and vals[row] <= 30.0:
            return 7 
        elif vals[row] > 30.0 and vals[row] <= 36.0:
            return 8
        elif vals[row] > 36.0 and vals[row] <= 44.0:
            return 9
        elif vals[row] > 44.0:
            return 10 

def get_vals(table, column_index, dim2 = False):
    vals = []
    final_vals = []
    for row in range(len(table)):
        vals.append(table[row][column_index])
    if dim2 == True:
        final_vals.append(vals)
        return final_vals
    return vals

def select_attribute(train, attributes):
    """ selects an attribute to pslit on based on smallest entropy
    Args:
        train: instances being looked at for entropy
        attributes: attributes that can be split on in the tree
    Returns:
        final attribute: attribut to split on with lowest entropy
    """
    # TODO: use entropy to compute and choose the attribute
    # with the smallest Enew
    E_final = {}
    curr_lowest = -1
    lowest_index = 0
    start_count = {}
    for row in train:
        if row[-1] not in start_count:
            start_count[row[-1]] = 0
    for attribute in attributes:
        E_counts = {}
        index = int(attribute[-1])
        for row in train:
            att_val = row[index]
            class_val = row[-1]
            if att_val in E_counts:
                E_counts[att_val][class_val] += 1
            else:
                E_counts[att_val] = start_count.copy()
                E_counts[att_val][class_val] += 1
        E_vals = []
        for val in E_counts:
            E_class = []
            total = 0
            for classifier in E_counts[val]:
                total += E_counts[val][classifier]
            for classifier in E_counts[val]:
                if E_counts[val][classifier] == 0:
                    E_value = 0
                else:
                    E_value = -E_counts[val][classifier] / total * np.log2(E_counts[val][classifier] / total)
                E_class.append(E_value)
            E_vals.append(np.sum(E_class) * total / len(train))
        E_new = np.sum(E_vals)
        if curr_lowest == -1:
            curr_lowest = E_new
            lowest_index = index
        elif curr_lowest > E_new:
            curr_lowest = E_new
            lowest_index = index
        E_final[attribute] = E_new
    final_attribute = "att" + str(lowest_index)
    # print("E_final:", E_final, "| Final attribute", final_attribute)
    return final_attribute

def partition_instances(train, instances, split_attribute):
    """ partitions the instances given to function based on split attribute
    Args:
        train: dataset
        isntances: instances being partitioned
        split_attribute: attribute to split on
    Returns:
        partiitions: dictionary of the partitions for the decision tree
    """
    partitions = {} 
    att_index = int(split_attribute[-1])
    attribute_domains = []
    for row in train:
        if row[att_index] not in attribute_domains:
            attribute_domains.append(row[att_index])
    for att_value in attribute_domains:
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)
    partitions = {key: val for key, val in sorted(partitions.items(), key = lambda ele: ele[0])}
    return partitions

def tdidt(train, instances, available_attributes, old_partition_length, F):
    """ creates a decision tree recursively from the dataset
    Args:
        train: dataset
        instances: terms being looked at in function
        available_attributes: attributes still available to split on
        old_partition_length: length of upper partition for Case 3
    Returns:
        tree: decision tree of instances 
    """
    F_attributes = random_attribute_subset(available_attributes, F)
    attribute = select_attribute(instances, F_attributes)
    # print("splitting on attribute:", attribute)
    available_attributes.remove(attribute)
    # print("Available Attribute", available_attributes)
    tree = ["Attribute", attribute]
    partitions = partition_instances(train, instances, attribute)
    # print("partitions:", partitions)
    partition_length = 0
    for att_value, att_partition in partitions.items():
        partition_length += len(att_partition)      
    for att_value, att_partition in partitions.items():
        # print("current attribute value:", att_value, len(att_partition))
        value_subtree = ["Value", att_value]
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            # print("CASE 1 all same class", tree)
            # TODO: make a leaf node
            leaf = ["Leaf", att_partition[0][-1], len(att_partition), partition_length]
            value_subtree.append(leaf)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            # print("CASE 2 no more attributes", tree)
            # TODO: we have a mix of labels, handle clash with majority
            # vote leaf node
            majority_value = find_majority_class(att_partition)
            leaf = ["Leaf", majority_value, len(att_partition), partition_length]
            value_subtree.append(leaf)
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            # print("CASE 3 empty partition", tree)
            new_partition = []
            for val in partitions:
                for instance in partitions[val]:
                    new_partition.append(instance)
            # print("New Partition:", new_partition)
            majority_value = find_majority_class(new_partition)
            tree = ["Leaf", majority_value, len(new_partition), old_partition_length]
            return tree
        else: # the previous conditions are all false... recurse!!
            # print("CASE RECURSION", tree)
            subtree = tdidt(train, att_partition, available_attributes.copy(), partition_length, F)
            # note the copy
            # TODO: append subtree to value_subtree and to tree
            value_subtree.append(subtree)
        tree.append(value_subtree)
        # print("Tree", tree)
        
    return tree

                
def all_same_class(partition):
    """ finds if all partition has same class
    Args:
        partition: terms to be looked at for the class count
    Returns:
        True if all same class and False otherwise
    """
    curr_class = partition[0][-1]
    for item in partition:
        if item[-1] != curr_class:
            return False
    return True                
        
def find_majority_class(partition):
    """ finds the majority class for the partition
    Args:
        partition: terms to be looked at for the class count
    Returns:
        majority_value(str): the majority value for the class of the terms
    """
    counts = {}
    majority_count = 0
    majority_value = ""
    for item in partition:
        val = item[-1]
        if val in counts:
            counts[val] += 1
        else:
            counts[val] = 1
    for value in counts:
        if counts[value] > majority_count:
            majority_count = counts[value]
            majority_value = value
        elif counts[value] == majority_count:
            if value < majority_value:
                majority_value = value
    return majority_value

def find_rules_recursive(tree, attributes, values, attribute_names, class_name):
    for i in range(len(tree)):
        value = tree[i]
        if value == "Attribute":
            attributes.append(tree[i+1])
            for j in range(2, len(tree)):
                find_rules_recursive(tree[j], attributes.copy(), values.copy(), attribute_names, class_name)
        elif value == "Value":
            values.append(tree[i+1])
            find_rules_recursive(tree[i+2], attributes.copy(), values.copy(), attribute_names, class_name)
        elif value == "Leaf":
            curr_rule = "IF "
            for attribute, val in zip(attributes, values):
                curr_index = int(attribute[-1])
                new_attribute = attribute_names[curr_index]
                curr_rule += new_attribute + " == " + val + " AND "
            curr_rule = curr_rule[:-4] + "THEN "
            curr_rule += class_name + " == " + tree[i+1]
            print(curr_rule)
            return
                
def random_attribute_subset(attributes, F):
    # shuffle and pick first F
    shuffled = attributes[:] # make a copy
    random.shuffle(shuffled)
    return shuffled[:F]

def Nmaxelements(list1, N):
    final_list = []
    
    for i in range(0, N): 
        max1 = 0
        max_index = 0
          
        for j in range(len(list1)):   
            if j not in final_list:  
                if list1[j] > max1:
                    max1 = list1[j]
                    max_index = j
        final_list.append(max_index)          
    return final_list

<<<<<<< HEAD
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
=======
def find_tree_max(tree, max_count, max_value):
    for i in range(len(tree)):
        value = tree[i]
        if value == "Attribute":
            for j in range(2, len(tree)):
                find_tree_max(tree[j], max_count, max_value)
        elif value == "Value":
            find_tree_max(tree[i+2], max_count, max_value)
        elif value == "Leaf":
            if tree[i+2] > max_count:
                max_value = tree[i+1]
                max_count = tree[i+2]
            return max_value
>>>>>>> d31637340e1af23cb9f31b0f5bab8b9f666ec3c6
