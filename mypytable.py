

##############################################
# Programmer: Aaron Miller
# Class: CptS 322-02, Spring 2021
# Programming Assignment #7
# 4/14/22
# 
# Description: This file houses the MyPyTable class which successfuclly handles functions for a table class like reading and writing to a file and deleting duplicates
##############################################
import copy
import csv

from numpy import append
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
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

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        rows = len(self.data)
        cols = len(self.data[0])
        
        return rows, cols # TODO: fix this

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            list of obj: 1D list of values in the column
        Notes:
            Raise ValueError on invalid col_identifier
        """
        values = []
        if include_missing_values == False:
            self.remove_rows_with_missing_values()
        index = -1
        try:
            for i in range(len(self.column_names)):
                if col_identifier == self.column_names[i]:
                    index = i;  
            if index == -1:
                raise ValueError   
        except ValueError:
            print("Invalid column identifier")
        if index != -1:
            for row in self.data:
                values.append(row[index])
        return values

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    numeric_value = float(self.data[i][j])
                    self.data[i][j] = numeric_value
                except ValueError:
                    pass
        return self
 

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.
        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        row_indexes_to_drop.sort()
        count = 0
        for index in row_indexes_to_drop:
            self.data.pop(index - count)
            count += 1
        return self
    
    def append_table(self, other_table):
        new_table = MyPyTable()
        new_table.data = copy.deepcopy(self.data)
        new_table.column_names = copy.deepcopy(self.column_names)
        for row in other_table.data:
            new_table.data.append(row)
        
        return new_table
            

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
        # TODO: finish this
        self.data = []
        infile = open(filename, "r")
        lines = infile.readlines()
        for line in lines:
            line = line.strip()
            values = line.split(",")
            self.data.append(values)
        infile.close()
        self.column_names = self.data[0]
        self.data.pop(0)
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

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
        duplicate_list = []
        check_list = []
       
        for row in self.data:
            add = True
            for item in check_list:
                for name in key_column_names:
                    index = self.column_names.index(name)
                    if item[index] == row[index]:
                        add = False
                        duplicate_list.append(self.data.index(row))
            if add == True:
                check_list.append(row)
        return duplicate_list 

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        data_copy = []
        add = 0
        for row in self.data:
            add = 0
            for item in row:
                if item == 'NA':
                    add = 1
            if add == 0:
                data_copy.append(row)
        self.data = copy.deepcopy(data_copy)
        return self

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        data_row = []
        index = self.column_names.index(col_name)
        for row in self.data:
            if row[index] != "NA":
                data_row.append(row[index])
        if len(data_row) > 0:
            avg = sum(data_row) / len(data_row)   
        for row in self.data:
            if row[index] == "NA":
                row[index] = avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        table = MyPyTable(col_names)
        for name in col_names:
            data_row = []
            index = self.column_names.index(name)
            for row in self.data:
                if row[index] != "NA":
                    data_row.append(row[index])
            if len(data_row) > 0:
                minimum = min(data_row)
                maximum = max(data_row)
                mid = (minimum + maximum) / 2
                avg = sum(data_row) / len(data_row)
                data_row = sorted(data_row) #len(data_row) // 2
                middle = len(data_row) // 2
                median = (data_row[middle] + data_row[~middle]) / 2
                new_row = [name, minimum, maximum, mid, avg, median]
                table.data.append(new_row)
             
        return table # TODO: fix this

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """
        new_table = MyPyTable()
         # build header
        new_column = []
        add_index_list = []
        for name in self.column_names:
            new_column.append(name)
        for name in other_table.column_names:
            if name not in key_column_names:
                add_index_list.append(other_table.column_names.index(name))
                new_column.append(name)
        new_table.column_names = new_column
        # join lists
        for row in self.data:
            for row2 in other_table.data:
                for name in key_column_names:
                    index = self.column_names.index(name)
                    index2 = other_table.column_names.index(name)
                    if row[index] == row2[index2]:
                        match = True
                    else:
                        match = False
                        break
                if match == True:
                    # join two rows
                    append_row = []
                    for item in row:
                        append_row.append(item)
                    for i in add_index_list:
                        append_row.append(row2[i])
                    new_table.data.append(append_row)
        return new_table 
    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """
        new_table = MyPyTable()
         # build header
        new_column = []
        add_index_list = []
        matched_index = []
        for name in self.column_names:
            new_column.append(name)
        for name in other_table.column_names:
            if name not in key_column_names:
                add_index_list.append(other_table.column_names.index(name))
                new_column.append(name)
        new_table.column_names = new_column
        # join lists
        index_joined_other = []
        index_joined = []
        for row in self.data:
            for row2 in other_table.data:
                for name in key_column_names:
                    index = self.column_names.index(name)
                    index2 = other_table.column_names.index(name)
                    if row[index] == row2[index2]:
                        match = True
                    else:
                        match = False
                        break
                if match == True:
                    # join two rows
                    matched_index.append(other_table.data.index(row2))
                    index_joined.append(self.data.index(row))
                    index_joined_other.append(other_table.data.index(row2))
                    append_row = []
                    for item in row:
                        append_row.append(item)
                    for i in add_index_list:
                        append_row.append(row2[i])
                    new_table.data.append(append_row)
        #Append other items
        for row in self.data:
            if self.data.index(row) not in index_joined:
                append_row = []
                for item in row:
                    append_row.append(item)
                for i in range(len(add_index_list)):
                    append_row.append("NA")
                new_table.data.append(append_row)
        for row in other_table.data:
            if other_table.data.index(row) in matched_index:
                continue
            append_row = ["NA"] * len(new_table.column_names)
            for col, item in enumerate(row):
                col2 = other_table.column_names[col]
                insert = new_table.column_names.index(col2)
                append_row[insert] = item
            new_table.data.append(append_row)
        return new_table
