import os
import re
import pickle
import argparse
import pandas as pd

from tools import load_abbrev_coltype, load_tbls_cols_types, load_table_datas, replace_comments

add_parser = argparse.ArgumentParser()
add_parser.add_argument('--db', type=str, default=None, help='database')
add_parser.add_argument('--bs', type=int, default=40, help='bin_size')
add_parser.add_argument('--usage', type=str, default=None, help='pretrain, finetune, test')

args = add_parser.parse_args()

current_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = f"{current_dir}/../../datas"

# load abbrev: table name and alias, col_type: continuous or discrete
abbrev, col_type = load_abbrev_coltype(folder_path, args.db, args.usage)
abbrev_inv = {v: k for k, v in abbrev.items()}

# load each table's column types
tbls_cols_types, decimal_tbls_cols = load_tbls_cols_types(folder_path, args.db)

# load data
print("------------------load table------------------")
tables = load_table_datas(folder_path, args.db, abbrev, tbls_cols_types)

# load hisgram
histogram_file = f"{folder_path}/statistics/{args.usage}/{args.db}/gen_histogram{args.bs}.pkl"
assert os.path.exists(histogram_file)
with open(histogram_file, 'rb') as f:
    histogram = pickle.load(f)

# load workload
workload_file = f"{folder_path}/workloads/{args.usage}/{args.db}/workloads.sql"
print(f"read workload: {workload_file}")
joins = set([])
with open(workload_file, 'r') as wf:
    for line in wf:
        line = replace_comments(line, '')
        line = line.split("||")[0].strip()
        if line.startswith('select'):
            if 'where' not in line:
                continue
            candidates = line.strip().split('where')[1].strip(';').strip()
            candidates = re.split(r'(?i)\band\b', candidates)        
        elif line.startswith('SELECT'):
            if 'WHERE' not in line:
                continue
            candidates = line.strip().split('WHERE')[1].strip(';').strip()
            candidates = re.split(r'(?i)\band\b', candidates)
        else:
            raise ValueError('workload file must start with select or SELECT')
        
        candidates = [c.strip('(') for c in candidates]
        candidates = [c.strip(')') for c in candidates]
        candidates = [c.strip() for c in candidates if ' = ' in c and '.' in c.split(' = ')[0] and '.' in c.split(' = ')[1]]
        for c in candidates:
            left, right = c.split('=')[0].strip(), c.split('=')[1].strip()
            left = left.replace('(', '').replace(')', '').replace(';', '')
            right = right.replace('(', '').replace(')', '').replace(';', '')
            if left.split('.')[0] in abbrev_inv and right.split('.')[0] in abbrev_inv:
                if (left, right) not in joins and (right, left) not in joins:
                    joins.add((left, right))
                else:
                    continue
            else:
                continue
print('joins:', joins)

fanout = {}
print('------------------calculating fanout...------------------')
for join in joins:
    left, right = join[0], join[1]
    left_table, left_column = left.split('.')[0], left.split('.')[1].replace(')', '').replace('(', '').replace(';', '')
    right_table, right_column = right.split('.')[0], right.split('.')[1].replace(')', '').replace('(', '').replace(';', '')
    print('left_table:', left_table, 'left_column:', left_column)
    print('right_table:', right_table, 'right_column:', right_column)
    
    value_counts = tables[right_table][right_column].value_counts()
    value_counts = pd.DataFrame(value_counts)
    table_merge = pd.merge(left=tables[left_table][left_column], right=value_counts, left_on=left_column, right_on=right_column, how='left')
    bin_edges = histogram[left_table][left_column]['bin_edges']
    left_fanout = []
    assert len(bin_edges) == args.bs + 1
    for i in range(len(bin_edges) - 1):
        if i != args.bs - 1:
            tmp = table_merge.loc[(table_merge[left_column] >= bin_edges[i]) & (table_merge[left_column] < bin_edges[i + 1]), 'count']
            tmp = tmp.sum()
            if tmp == 0:
                pass
            else:
                tmp = tmp / len(table_merge.loc[(table_merge[left_column] >= bin_edges[i]) & (table_merge[left_column] < bin_edges[i + 1])])
                assert len(table_merge.loc[(table_merge[left_column] >= bin_edges[i]) & (table_merge[left_column] < bin_edges[i + 1])]) != 0
        else:
            tmp = table_merge.loc[(table_merge[left_column] >= bin_edges[i]) & (table_merge[left_column] <= bin_edges[i + 1]), 'count']
            tmp = tmp.sum()
            if tmp == 0:
                pass
            else:
                tmp = tmp / len(table_merge.loc[(table_merge[left_column] >= bin_edges[i]) & (table_merge[left_column] <= bin_edges[i + 1])])
                assert len(table_merge.loc[(table_merge[left_column] >= bin_edges[i]) & (table_merge[left_column] <= bin_edges[i + 1])]) != 0
        left_fanout.append(tmp)
    
    right_fanout = []
    value_counts = tables[left_table][left_column].value_counts()
    value_counts = pd.DataFrame(value_counts)
    table_merge = pd.merge(left=tables[right_table][right_column], right=value_counts, left_on=right_column, right_on=left_column, how='left')
    bin_edges = histogram[right_table][right_column]['bin_edges']
    assert len(bin_edges) == args.bs + 1
    for i in range(len(bin_edges) - 1):
        if i != args.bs - 1:
            tmp = table_merge.loc[(table_merge[right_column] >= bin_edges[i]) & (table_merge[right_column] < bin_edges[i + 1]), 'count']
            tmp = tmp.sum()
            if tmp == 0:
                pass
            else:
                tmp = tmp / len(table_merge.loc[(table_merge[right_column] >= bin_edges[i]) & (table_merge[right_column] < bin_edges[i + 1])])
                assert len(table_merge.loc[(table_merge[right_column] >= bin_edges[i]) & (table_merge[right_column] < bin_edges[i + 1])]) != 0
        else:
            tmp = table_merge.loc[(table_merge[right_column] >= bin_edges[i]) & (table_merge[right_column] <= bin_edges[i + 1]), 'count']
            tmp = tmp.sum()
            if tmp == 0:
                pass
            else:
                tmp = tmp / len(table_merge.loc[(table_merge[right_column] >= bin_edges[i]) & (table_merge[right_column] <= bin_edges[i + 1])])
                assert len(table_merge.loc[(table_merge[right_column] >= bin_edges[i]) & (table_merge[right_column] <= bin_edges[i + 1])]) != 0
        right_fanout.append(tmp)
        
    fanout[join] = [left_fanout, right_fanout]
    join_inv = (join[1], join[0])
    fanout[join_inv] = [right_fanout, left_fanout]
    print('join:', join)
    print('left_bin_edges:', histogram[left_table][left_column]['bin_edges'])
    print('left_fanout:', left_fanout)
    print('right_bin_edges:', histogram[right_table][right_column]['bin_edges'])
    print('right_fanout:', right_fanout)

output_file = f"{folder_path}/statistics/{args.usage}/{args.db}/gen_fanout{args.bs}.pkl"
print('output file path:', output_file)
if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))
with open(output_file, 'wb') as f:
    pickle.dump(fanout, f)

print('------------------calculating fanout done------------------')
