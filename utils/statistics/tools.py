import os
import re
import pickle
import numpy as np
import pandas as pd

def replace_comments(text, replacement):
    pattern = r"/\*.*?\*/"
    modified_text = re.sub(pattern, replacement, text)
    return modified_text

def load_abbrev_coltype(folder_path, db, usage):
    col_type_path = f"{folder_path}/statistics/{usage}/{db}/abbrev_col_type.pkl"
    assert os.path.exists(col_type_path), f'{col_type_path} not exists'
    with open(col_type_path, 'rb') as f:
        data = pickle.load(f)
        abbrev = data['abbrev']
        col_type = data['col_type']
        print('abbrev: ', abbrev)
    for table in col_type:
        for c_type in col_type[table]:
            print(f"{table}.{c_type}: {col_type[table][c_type]}")
        print('-' * 50)
    print('-' * 100)
    return abbrev, col_type

def load_tbls_cols_types(folder_path, db):
    type_file = f"{folder_path}/datasets/{db}/postgres_create_{db}.sql"
    print(f'load col type file: {type_file}')
    assert os.path.exists(type_file)
    with open(type_file, 'r') as file:
        tbls_cols_types, decimal_tbls_cols = {}, {}
        with open(type_file, 'r') as file:
            for line in file:
                if "create table" in line:
                    tbl = line.split("create table")[1].split()[0].replace('"', '')
                    tbls_cols_types[tbl] = {}
                    decimal_tbls_cols[tbl] = []
                elif "integer" in line or "bigint" in line or "smallint" in line:
                    col = line.strip().split(" ")[0].replace('"', '')
                    tbls_cols_types[tbl][col] = pd.Int64Dtype()
                elif "character" in line or "varchar(" in line or "char(" in line: 
                    col = line.strip().split(" ")[0].replace('"', '')
                    tbls_cols_types[tbl][col] = pd.StringDtype()
                elif "decimal(" in line:
                    col = line.strip().split(" ")[0].replace('"', '')
                    decimal_tbls_cols[tbl].append(col)
                elif "double precision" in line:
                    col = line.strip().split(" ")[0].replace('"', '')
                    tbls_cols_types[tbl][col] = pd.Float64Dtype()
                else:
                    pass
    print(tbls_cols_types)     
    return tbls_cols_types, decimal_tbls_cols

def load_table_datas(folder_path, db, abbrev, tbls_cols_types):
    load_file = f"{folder_path}/datasets/{db}/postgres_create_{db}.sql"
    tables = {}
    print("load table info from " + load_file)
    with open(load_file, 'r') as lf:
        for line in lf:
            if line.startswith('\copy') or line.startswith('\COPY'):
                tablename = line.split(' ')[1].strip("'")
                filename = line.split(' ')[3].strip("'")
                path = f"{folder_path}/datasets/{db}/{filename}"
                assert os.path.exists(path)
                table = pd.read_csv(path, sep='|', quotechar='"', escapechar='\\', dtype=tbls_cols_types[tablename], keep_default_na=False, na_values=['NULL'])  
                
                assert tablename not in tables
                assert tablename in abbrev

                print('load table: ', tablename, ' as ', abbrev[tablename])
                tables[abbrev[tablename]] = table
    return tables

def get_histogram(tables, table, column, bin_size):
    len_column = tables[table][column].shape[0]
    tmp_column = tables[table][column].dropna()
    print('type of tmp_column: ', type(tmp_column))
    print('tmp_column: ', tmp_column)
    tmp_column = tmp_column.astype(float)
    if len(tmp_column) == 0:
        return np.zeros(bin_size), np.zeros(bin_size), 0, 0, 0
    hist, bin_edges = np.histogram(tmp_column, bins=bin_size, density=False)
    hist = hist.astype(float)
    min_value, max_value = min(tmp_column), max(tmp_column)
    return hist, bin_edges, len_column, min_value, max_value

def get_summary(tables, table, column):
    """ space saving summary algorithm"""
    K = 99999999
    summary = {}
    tmp_column = tables[table][column].dropna()
    print('current column:', column)
    for c in tmp_column:
        if c in summary:
            summary[c] += 1
        elif len(summary) < K:
            summary[c] = 1
        else:
            raise ValueError('summary length must be less than K')
    keys, values = list(summary.keys()), list(summary.values())
    assert len(keys) == len(values)
    return keys, values
