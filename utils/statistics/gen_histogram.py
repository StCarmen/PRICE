import os
import pickle
import argparse

from tools import load_abbrev_coltype, load_tbls_cols_types, load_table_datas, get_histogram

add_parser = argparse.ArgumentParser()
add_parser.add_argument('--db', type=str, default=None, help='database')
add_parser.add_argument('--bs', type=int, default=40, help='bin_size')
add_parser.add_argument('--usage', type=str, default=None, help='pretrain, finetune, test')

args = add_parser.parse_args()

current_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = f"{current_dir}/../../datas"

results = {}

# load abbrev: table name and alias, col_type: continuous or discrete
abbrev, col_type = load_abbrev_coltype(folder_path, args.db, args.usage)

# load each table's column types
tbls_cols_types, decimal_tbls_cols = load_tbls_cols_types(folder_path, args.db)

# load data
print("------------------load table------------------")
tables = load_table_datas(folder_path, args.db, abbrev, tbls_cols_types)

print('------------------calculating histogram...------------------')
for table in tables:
    results[table] = {}
    for column in tables[table].columns:
        if column in col_type[table]['ctn']:
            hist, bin_edges, len_col, min_value, max_value = get_histogram(tables, table, column, args.bs)
            print(f"{table}.{column}: hist:{hist}  bin_edges:{bin_edges}  len:{len_col}  min:{min_value}  max:{max_value}\n\n")
            results[table][column] = {'hist':hist, 'bin_edges':bin_edges, 'len':len_col, 'min_value':min_value, 'max_value':max_value}
        else:
            pass

print(f"results: {results}")

output_file = f"{folder_path}/statistics/{args.usage}/{args.db}/gen_histogram{args.bs}.pkl"
print('output file path:', output_file)            
if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))
with open(output_file, 'wb') as f:
    pickle.dump(results, f)

print('------------------calculating histogram done------------------')
