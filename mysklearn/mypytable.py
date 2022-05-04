import copy
import csv

#from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        max_row = 0
        max_col = 0
        for row in self.data:
            curr_col = 0
            max_row += 1
            for _ in row:
                curr_col += 1
            if curr_col >= max_col:
                max_col = curr_col
        return max_row, max_col

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ('NA' or '')
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        # parse the input
        if type(col_identifier) is str:
            index = self.column_names.index(col_identifier)
        elif type(col_identifier) is int:
            index = col_identifier
        else:
            print('Invalid type for column identifier')
            raise TypeError
        # parse the data for the column
        try:
            new_col = []
            for row in self.data:
                if include_missing_values or (row[index] != 'NA' and row[index] !=''):
                    new_col.append(row[index])
            return new_col
        except IndexError:
            print('Invalid column identifier.')
            raise ValueError

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numferic.
        """
        # go val by val try to convert to float
        for row_index, row_value in enumerate(self.data):
            for col_index, col_value in enumerate(row_value):
                # try to convert, if fail do nothing
                try:
                    self.data[row_index][col_index] = float(col_value)
                except:
                    continue

    def convert_to_string(self):
        """Try to convert each value in the table to a string type (str).

        Notes:
            Leave values as is that cannot be converted to numferic.
        """
        # go val by val try to convert to str
        for row_index, row_value in enumerate(self.data):
            for col_index, col_value in enumerate(row_value):
                self.data[row_index][col_index] = str(col_value)

        
    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        # remove back to front to avoid changing indexes
        no_duplicates = set(row_indexes_to_drop)
        indices = list(no_duplicates)
        sorted_indices = sorted(indices)
        for index in sorted_indices[::-1]:
            try:
                del self.data[index]
            except Exception as e:
                print(e, index)
                continue
                

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            self.column_names = next(reader)
            for row in reader:
                self.data.append(row)
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w+') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.column_names) # write one header row
            writer.writerows(self.data) # write the rest of the data

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        # make sure the columns are valid
        for key in key_column_names:
            if key not in self.column_names:
                return []
        duplicates = [] # indexes of duplicates
        key_column_vals = [] # keep a running count of key values
        # find values of key indices, compare against key_column_values
        for index, row in enumerate(self.data):        
            row_values = []          
            for key in key_column_names:
                key_index = self.column_names.index(key)  
                row_values.append(row[key_index])
            if row_values in key_column_vals and index not in duplicates:
                duplicates.append(index)
            else:
                key_column_vals.append(row_values)
        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ('NA' or '').
        """
        missing_values = []
        # go through looking for 'NA'
        for index, row in enumerate(self.data):
            for val in row:
                if val == '' or val == 'NA':
                    missing_values.append(index)
        # use drop_rows call to avoid writing more
        self.drop_rows(missing_values)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        try:
            key_index = self.column_names.index(col_name)
            column = self.get_column(col_name)
        except:
            print('Invalid column identifier')
            return
        total = 0
        num = 0
        average = 0
        for val in column:
            try:
                total += val
                num += 1
            except:
                continue
        if not num:
            return
        average = total/num
        # look for missing data
        for row in self.data:
            if row[key_index] == 'NA' or '':
                row[key_index] = average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        # placeholders for header and body of new table
        summary_columns = ["attribute", "min", "max", "mid", "avg", "median"]
        summary_data = []
        # go through every column and calc its stats
        for col in col_names:
            column_data = sorted(self.get_column(col, False))
            length = len(column_data)
            try:
                odd = length % 2
                minimum = column_data[0]
                maximum = column_data[-1]
                middle = (abs(column_data[-1])-abs(column_data[0]))/2
                average = sum(column_data) / len(column_data)
            except:
                continue
            # calculate median differently if odd or even
            if odd:
                median = column_data[length//2]
            else:
                median = (column_data[length//2-1]+column_data[(length//2)])*0.5
            summary_data.append([col, minimum, maximum, middle, average, median])
        return MyPyTable(summary_columns, summary_data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # init some vars to hold new table data
        new_data = []
        new_header = self.column_names
        for col_name in other_table.column_names:
            if col_name not in new_header:
                new_header.append(col_name)
        # for row in self, look through the other data to find a row that matches on all key col
        for row in self.data:
            for other_row in other_table.data:
                match = True
                # look for row that matches all key_column_names
                for col in key_column_names:
                    if row[self.column_names.index(col)] != other_row[other_table.column_names.index(col)]:
                        match = False
                        break
                # if match, combine the two rows
                if match:
                    new_row = ['NA']*len(new_header)
                    for index, value in enumerate(row):
                        new_row[index] = value
                    for index, value in enumerate(other_row):
                        other_column = other_table.column_names[index]
                        if other_column in key_column_names:
                            continue
                        insert_index = new_header.index(other_column)
                        new_row[insert_index] = value
                    new_data.append(new_row)
        return MyPyTable(new_header, new_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with 'NA' or ''.
        """
        # init some vars to hold new table data
        new_data = []
        new_header = self.column_names
        for col_name in other_table.column_names:
            if col_name not in new_header:
                new_header.append(col_name)
        # for row in self, look through the other data to find a row that matches on all key col
        matched_indices = []
        for row in self.data:
            one_match = False # track if this row has gotten 1+ matches
            for other_row_index, other_row in enumerate(other_table.data):
                match = True
                # look for row that matches all key_column_names
                for col in key_column_names:
                    if row[self.column_names.index(col)] != other_row[other_table.column_names.index(col)]:
                        match = False
                        break
                new_row = ['NA']*len(new_header)
                for index, value in enumerate(row):
                    new_row[index] = value
                # if match found in other table
                if match:
                    one_match = True
                    matched_indices.append(other_row_index)
                    for index, value in enumerate(other_row):
                        other_column = other_table.column_names[index]
                        if other_column in key_column_names:
                            continue
                        insert_index = new_header.index(other_column)
                        new_row[insert_index] = value
                    new_data.append(new_row)
            # if self has no matches in other table, add it with NA
            if not one_match:
                new_data.append(new_row)
        # go through other table for every row without a match
        for index, row in enumerate(other_table.data):
            if index in matched_indices:
                continue
            new_row = ['NA'] * len(new_header)
            for col, value in enumerate(row):
                other_column = other_table.column_names[col]
                insert_index = new_header.index(other_column)
                new_row[insert_index] = value
            new_data.append(new_row)
        return MyPyTable(new_header, new_data)
