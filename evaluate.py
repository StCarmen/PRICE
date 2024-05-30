import os
import datetime

import torch
import numpy as np
import torch.nn as nn

from model.encoder import RegressionModel
from utils.model.dataset import load_dataset_features, make_feature_datasets, make_test_feature_dataloaders
from utils.model.padding import features_padding
from utils.model.perror_input import generate_perror_input
from utils.model.qerror import get_qerror, interval_qerror
from utils.model.args import get_args

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

TEST_LIST = ['imdb', 'stats', 'ergastf1', 'genome']

args = get_args()
print(args)
current_dir = os.path.dirname(os.path.abspath(__file__))

test_data, test_labels, test_pg_est_cards, \
test_n_join_cols, test_n_fanouts, test_n_tables, test_n_filter_cols, test_list_lens = load_dataset_features(bin_size=args.bin_size, dataset_list=TEST_LIST, train_or_test='test', usage='test')

max_n_join_col, max_n_fanout, max_n_table, max_n_filter_col = max(test_n_join_cols), max(test_n_fanouts), max(test_n_tables), max(test_n_filter_cols)
test_data, test_padding_masks = features_padding(args.bin_size, args.table_dim, args.filter_dim,
                                             test_data, test_n_join_cols, test_n_fanouts, test_n_tables, test_n_filter_cols, 
                                             max_n_join_col, max_n_fanout, max_n_table, max_n_filter_col)
print("dataset padding done!!")
test_datasets_list = make_feature_datasets(test_data, test_labels, test_pg_est_cards, test_padding_masks,
                                      test_n_join_cols, test_n_fanouts, test_n_tables, test_n_filter_cols,
                                      train_or_test='test', test_list_lens=test_list_lens)
test_loaders_list = make_test_feature_dataloaders(test_datasets_list, test_list_lens)

# our model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegressionModel(n_join_col=max_n_join_col, n_fanout=max_n_fanout, n_table=max_n_table, n_filter_col=max_n_filter_col,
                        hist_dim=args.bin_size, table_dim=args.table_dim, filter_dim=args.filter_dim,
                        query_hidden_dim=args.query_hidden_dim, final_hidden_dim=args.final_hidden_dim, output_dim=args.output_dim,
                        n_embd=args.n_embd, n_layers=args.n_layers, n_heads=args.n_heads, dropout_rate=args.dropout_rate).to(device)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

model_path = f'{current_dir}/results/model_params.pth'
print(f"load model from {model_path}")
model.load_state_dict(torch.load(model_path))

criterion = nn.MSELoss()

print('--'*30)
model.eval()
for idx, current_dataloader in enumerate(test_loaders_list):
    test_loss = 0
    for i, (data, label, pg_est_card, padding_mask, n_join_col, n_fanout, n_table, n_filter_col) in enumerate(current_dataloader):
        data = data.to(torch.float).to(device)
        n_join_col = n_join_col.to(torch.float).to(device).view(-1, 1)
        n_fanout = n_fanout.to(torch.float).to(device).view(-1, 1)
        n_table = n_table.to(torch.float).to(device).view(-1, 1)
        n_filter_col = n_filter_col.to(torch.float).to(device).view(-1, 1)
        pg_est_card = pg_est_card.to(torch.float).to(device).view(-1, 1)
        pg_est_card = torch.log(pg_est_card + 1) + 1
        label = torch.log(label.to(torch.float).to(device) + 1) + 1
        label = label.view(1, -1)

        with torch.no_grad():
            output = model(data, pg_est_card, padding_mask, n_join_col, n_fanout, n_table, n_filter_col).view(1, -1)
            loss = criterion(output, label)
            test_loss += loss.item() * len(data)
    test_loss = test_loss / len(current_dataloader.dataset)
    print(f"{TEST_LIST[idx]} loss: {test_loss}")
    q_error = get_qerror(output, label, cuda=True, do_scale=True, percentile_list=[30, 50, 80, 90, 95, 99])
    print(f'{TEST_LIST[idx]} q-error: 30%:', q_error[0], '  50%:', q_error[1], '  80%:', q_error[2], '  90%:', q_error[3], '  95%:', q_error[4], '  99%:', q_error[5])
    interval_qerror(output, label, cuda=True, do_scale=True)

    # to generate p-error input file
    output1 = output[0].detach().cpu().numpy()
    workloads_test_file_path = f'{current_dir}/datas/workloads/test/{TEST_LIST[idx]}/workloads.sql'
    workloads_all_file_path = f'{current_dir}/datas/workloads/test/{TEST_LIST[idx]}/workloads_all.sql'
    out_path = f'{current_dir}/results/{TEST_LIST[idx]}_perror_input.sql'
    generate_perror_input(output1, out_path, workloads_test_file_path, workloads_all_file_path, True)

print('done!')
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
