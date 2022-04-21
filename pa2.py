##############################################
# Programmer: Aaron Miller
# Class: CptS 322-02, Spring 2021
# Programming Assignment #2
# 2/9/22
# 
# Description: This program houses functions that perform multiple functions on multiple tables and files layed out in the assignment 
##############################################

"""
 I fixed some mispellings during cleaning like a mispelled toyota and deleted instances where there was a match but there wasn't mpg data but there was a price.
"""

import os

from mypytable import MyPyTable

def print_divider():
    print("----------------------------------")
def print_duplicate_data(filename, table, key_names):
    print_divider()
    print(f'{filename}:')
    print_divider
    print("Number of instances: ", len(table.data))
    duplicate_num = len(table.find_duplicates(key_names))
    print("Duplicates:", duplicate_num)
    
def print_duplicate_removed(filename, table, key_names):
    print_divider()
    print(f"duplicates removed (saved as {filename}):")
    print_divider
    print("Number of instances: ", len(table.data))
    duplicates = table.find_duplicates(key_names)
    index = duplicates[-1]
    print("Duplicates:", table.data[index])
    table.drop_rows(duplicates)
    table.save_to_file(filename)
    
def print_cleaned(filename, table, key_names):
    print_divider()
    print(f'{filename}:')
    print_divider
    print("Number of instances: ", len(table.data))
    duplicates = table.find_duplicates(key_names)
    duplicates_list = []
    for item in duplicates:
        duplicates_list.append(table.data[item])
    print("Duplicates:", duplicates_list)

def join_tables(filename, table1, table2, key_names):
    joined_table = table1.perform_full_outer_join(table2, key_names)
    joined_table.save_to_file(filename)
    print_divider()
    print(f"combined table (saved as {filename}):")
    print("Number of instances: ", len(joined_table.data))
    duplicates = joined_table.find_duplicates(key_names)
    duplicates_list = []
    for item in duplicates:
        duplicates_list.append(joined_table.data[item])
    print("Duplicates:", duplicates_list)
    
def summary_stats(filename, key_names):
    table = MyPyTable()
    table.load_from_file(filename)
    stats_table = table.compute_summary_statistics(key_names)
    stats_table.pretty_print()

def last_step(filename1, filename2, key_names):
    table = MyPyTable()
    table.load_from_file(filename1)
    print_divider()


def main():
    mpg_fname = os.path.join("input_data", "auto-mpg.txt")
    prices_fname = os.path.join("input_data", "auto-prices.txt")
    print(mpg_fname, prices_fname)
    mpg_table = MyPyTable().load_from_file(mpg_fname)
    price_table = MyPyTable().load_from_file(prices_fname)
    print_duplicate_data(mpg_fname, mpg_table, ['car name', 'model year'])
    print_duplicate_data(prices_fname, price_table, ['car name', 'model year'])
    prices_output = os.path.join("output_data", "auto-prices-nodups.txt")
    mpg_output = os.path.join("output_data", "auto-mpg-nodups.txt")
    print_duplicate_removed(mpg_output, mpg_table, ['car name', 'model year'])
    print_duplicate_removed(prices_output, price_table, ['car name', 'model year'])
    prices_cleaned = os.path.join("output_data", "auto-prices-cleaned.txt")
    mpg_cleaned = os.path.join("output_data", "auto-mpg-cleaned.txt")
    print_cleaned(mpg_cleaned, mpg_table, ['car name', 'model year'])
    print_cleaned(prices_cleaned, price_table, ['car name', 'model year'])
    # join time
    joined_data = os.path.join("output_data", "auto-data.txt")
    join_tables(joined_data, mpg_table, price_table, ['car name', 'model year'])
    # summary stats
    summary_stats(joined_data, ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'msrp'])
    
if __name__ == "__main__":
    main()
