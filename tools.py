from prettytable import PrettyTable
import pandas as pd


def print_table(df: pd.DataFrame, rows_count: int = 10):
    table = PrettyTable()
    table.field_names = list(df.columns)
    for i, row in df.iloc[:rows_count, :].iterrows():
        table.add_row(row)
    print(table)
