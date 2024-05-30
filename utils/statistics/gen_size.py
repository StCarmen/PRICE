import os
import pickle
import argparse

from tools import load_abbrev_coltype, load_tbls_cols_types, load_table_datas

add_parser = argparse.ArgumentParser()
add_parser.add_argument('--db', type=str, default=None, help='database')
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

# get size of each table
for table in tables:
    results[table] = {}
    results[table]['size'] = tables[table].shape[0]
    results[table]['num_cols'] = tables[table].shape[1]
    results[table]['num_rows'] = tables[table].shape[0]

print(f"results: {results}")

output_file = f"{folder_path}/statistics/{args.usage}/{args.db}/gen_size.pkl"
print('output file path:', output_file)
if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))
with open(output_file, 'wb') as f:
    pickle.dump(results, f)

print('------------------get size done------------------')
