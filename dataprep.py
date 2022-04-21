
from sqlalchemy import tablesample
from mypytable import MyPyTable

def main():
    filename = ("PlayerData/playerstats2022norm.csv")
    norm_table_2022 = MyPyTable().load_from_file(filename)
    filename = ("PlayerData/playerstats2022adv.csv")
    adv_table_2022 = MyPyTable().load_from_file(filename)
    table_2022 = norm_table_2022.perform_full_outer_join(adv_table_2022, ["Player"])
    table_2022.remove_rows_with_missing_values()
    indexes = table_2022.find_duplicates(["Player"])
    table_2022.drop_rows(indexes)
    
    
    filename = ("PlayerData/playerstats2021norm.csv")
    norm_table_2021 = MyPyTable().load_from_file(filename)
    filename = ("PlayerData/playerstats2021adv.csv")
    adv_table_2021 = MyPyTable().load_from_file(filename)
    table_2021 = norm_table_2021.perform_full_outer_join(adv_table_2021, ["Player"])
    table_2021.remove_rows_with_missing_values()
    indexes = table_2021.find_duplicates(["Player"])
    table_2021.drop_rows(indexes)

    
    filename = ("PlayerData/playerstats2020norm.csv")
    norm_table_2020 = MyPyTable().load_from_file(filename)
    filename = ("PlayerData/playerstats2020adv.csv")
    adv_table_2020 = MyPyTable().load_from_file(filename)
    table_2020 = norm_table_2020.perform_full_outer_join(adv_table_2020, ["Player"])
    table_2020.remove_rows_with_missing_values()
    indexes = table_2020.find_duplicates(["Player"])
    table_2020.drop_rows(indexes)
    
    
    filename = ("PlayerData/playerstats2019norm.csv")
    norm_table_2019 = MyPyTable().load_from_file(filename)
    filename = ("PlayerData/playerstats2019adv.csv")
    adv_table_2019 = MyPyTable().load_from_file(filename)
    table_2019 = norm_table_2019.perform_full_outer_join(adv_table_2019, ["Player"])
    table_2019.remove_rows_with_missing_values()
    indexes = table_2019.find_duplicates(["Player"])
    table_2019.drop_rows(indexes)
    
    filename = ("PlayerData/playerstats2018norm.csv")
    norm_table_2018 = MyPyTable().load_from_file(filename)
    filename = ("PlayerData/playerstats2018adv.csv")
    adv_table_2018 = MyPyTable().load_from_file(filename)
    table_2018 = norm_table_2018.perform_full_outer_join(adv_table_2018, ["Player"])
    table_2018.remove_rows_with_missing_values()
    indexes = table_2018.find_duplicates(["Player"])
    table_2018.drop_rows(indexes)
    
    # make mega table
    table = table_2022.append_table(table_2021)
    table = table.append_table(table_2020)
    table = table.append_table(table_2019)
    table = table.append_table(table_2018)
    
    # discretize data based on minutes played and games played
    indexes = []
    for index in range(len(table.data)):
        if table.data[index][1] < 35:
            indexes.append(index)
    table.drop_rows(indexes)
    
    indexes = []
    for index in range(len(table.data)):
        if table.data[index][2] < 28.9:
            indexes.append(index)
    table.drop_rows(indexes)
    
    table.column_names.append("All-star")
    for row in table.data:
        row.append("no")
    
    filename = "AllStarData.csv"
    table.save_to_file(filename)
    

    
if __name__ == "__main__":
    main()