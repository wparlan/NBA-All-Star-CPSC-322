import numpy as np

def get_frequencies(table, col_name):
    """
    Computes the frequency of each value in a given column

    Args:
        table(MyPyTable): MyPyTable of data
        col_name(str): name of the column to calculate frequencies on

    Returns:
        values: list of values. Parallel to counts.
        counts: list of counts for each value. Parallel to values.
    """
    col = table.get_column(col_name)
    try:
        table.convert_to_numeric()
        col.sort() # inplace
    except TypeError as e:
        print(e)     
        table.convert_to_string()
        col = table.get_column(col_name)
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

def get_frequencies_of_list(values):
    values.sort() # inplace
    # parallel lists
    freqs = []
    counts = []
    for value in values:
        if value in freqs: # seen it before
            counts[-1] += 1 # okay because sorted
        else: # haven't seen it before
            freqs.append(value)
            counts.append(1)

    return freqs, counts


def compute_bin_frequencies(values, cutoffs):
    """
    Computes frequencies of values using bins.

    Args:
        values(list): List of values to count
        cutoffs(list): List of cutoff points for bins

    Returns:
        list(int) of frequency of values in each bin
    """ 
    freqs = [0 for _ in range(len(cutoffs) - 1)] # because N + 1 cutoffs
    for value in values:
        if value == max(values):
            freqs[-1] += 1 # add one to the last bin count
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= value < cutoffs[i + 1]:
                    freqs[i] += 1 
                    # add one to this bin defined by [cutoffs[i], cutoffs[i+1])
    return freqs

def compute_equal_width_cutoffs(values, num_bins):
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
    
def get_percentage(table, columns):
    table.convert_to_numeric()
    data = []
    for col in columns:
        sum = 0
        column_data = table.get_column(col)
        for item in column_data:
            try:
                sum += float(item)
            except:
                continue
        data.append(sum)
    return data
    
def compute_covariance(x, y):
    # Finding the mean of the series x and y
    mean_x = sum(x)/float(len(x))
    mean_y = sum(y)/float(len(y))
    # Subtracting mean from the individual elements
    sub_x = [i - mean_x for i in x]
    sub_y = [i - mean_y for i in y]
    numerator = sum([sub_x[i]*sub_y[i] for i in range(len(sub_x))])
    denominator = len(x)-1
    cov = numerator/denominator
    return cov

def compute_correlation(x, y):
    # Finding the mean of the series x and y
    mean_x = sum(x)/float(len(x))
    mean_y = sum(y)/float(len(y))
    # Subtracting mean from the individual elements
    sub_x = [i-mean_x for i in x]
    sub_y = [i-mean_y for i in y]
    # covariance for x and y
    numerator = sum([sub_x[i]*sub_y[i] for i in range(len(sub_x))])
    # Standard Deviation of x and y
    std_deviation_x = sum([sub_x[i]**2.0 for i in range(len(sub_x))])
    std_deviation_y = sum([sub_y[i]**2.0 for i in range(len(sub_y))])
    # squaring by 0.5 to find the square root
    denominator = (std_deviation_x*std_deviation_y)**0.5 # same as sqrt each var
    cor = numerator/denominator
    return cor

def compute_slope_intercept(x, y):
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    num = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))])
    den = sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    m = num / den 
    # y = mx + b => b = y - mx 
    b = mean_y - m * mean_x
    return m, b 

def compute_ratings(table, column, cutoffs):
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