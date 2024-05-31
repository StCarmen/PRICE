import os
import datetime

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from model.encoder import RegressionModel
from utils.model.dataset import load_dataset_features, make_feature_datasets, make_train_feature_dataloaders
from utils.model.padding import features_padding
from utils.model.qerror import get_qerror
from utils.model.args import get_args

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

TRAIN_LIST = ['accidents', 'airline', 'baseball', 'basketball', 'carcinogenesis', 'ccs', 'chembl', 'consumer',
              'credit', 'employee', 'financial', 'fnhk', 'grants', 'hepatitis', 'hockey', 'legalacts', 'movielens',
              'sakila', 'sap', 'seznam', 'ssb', 'talkingdata', 'telstra', 'tournament', 'tpc_h', 'tubepricing']

args = get_args()
print(args)
current_dir = os.path.dirname(os.path.abspath(__file__))

train_data, train_labels, train_pg_est_cards, \
train_n_join_cols, train_n_fanouts, train_n_tables, train_n_filter_cols = load_dataset_features(bin_size=args.bin_size, dataset_list=TRAIN_LIST, train_or_test='train', usage='pretrain')

max_n_join_col, max_n_fanout, max_n_table, max_n_filter_col = max(train_n_join_cols), max(train_n_fanouts), max(train_n_tables), max(train_n_filter_cols)
train_data, train_padding_masks = features_padding(args.bin_size, args.table_dim, args.filter_dim,
                                                   train_data, train_n_join_cols, train_n_fanouts, train_n_tables, train_n_filter_cols,
                                                   max_n_join_col, max_n_fanout, max_n_table, max_n_filter_col)
print("dataset padding done!!")
train_dataset = make_feature_datasets(train_data, train_labels, train_pg_est_cards, train_padding_masks,
                                      train_n_join_cols, train_n_fanouts, train_n_tables, train_n_filter_cols,
                                      train_or_test='train')
train_loader = make_train_feature_dataloaders(train_dataset, args.batch_size)

# our model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegressionModel(n_join_col=max_n_join_col, n_fanout=max_n_fanout, n_table=max_n_table, n_filter_col=max_n_filter_col,
                        hist_dim=args.bin_size, table_dim=args.table_dim, filter_dim=args.filter_dim,
                        query_hidden_dim=args.query_hidden_dim, final_hidden_dim=args.final_hidden_dim, output_dim=args.output_dim,
                        n_embd=args.n_embd, n_layers=args.n_layers, n_heads=args.n_heads, dropout_rate=args.dropout_rate).to(device)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
criterion = nn.MSELoss()

for epoch in range(args.epochs):
    print('--'*30)
    model.train()
    train_loss = 0
    all_output, all_label = [], []
    for i, (feature, label, pg_est_card, padding_mask, n_join_col, n_fanout, n_table, n_filter_col) in enumerate(train_loader):
        feature = feature.to(torch.float).to(device)
        n_join_col = n_join_col.to(torch.float).to(device).view(-1, 1)
        n_fanout = n_fanout.to(torch.float).to(device).view(-1, 1)
        n_table = n_table.to(torch.float).to(device).view(-1, 1)
        n_filter_col = n_filter_col.to(torch.float).to(device).view(-1, 1)
        pg_est_card = pg_est_card.to(torch.float).to(device).view(-1, 1)
        pg_est_card = torch.log(pg_est_card + 1) + 1
        label = torch.log(label.to(torch.float).to(device) + 1) + 1
        label = label.view(1, -1)

        optimizer.zero_grad()
        output = model(feature, pg_est_card, padding_mask, n_join_col, n_fanout, n_table, n_filter_col).view(1, -1)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(feature)
        all_output.extend(output.cpu().data.tolist()[0])
        all_label.extend(label.cpu().data.tolist()[0])

    scheduler.step()
    train_loss = train_loss / len(train_loader.dataset)
    print(f"epoch: {epoch}, train loss: {train_loss}")

    all_output, all_label = np.array(all_output), np.array(all_label)
    q_error = get_qerror(all_output, all_label, cuda=False, do_scale=True, percentile_list=[30, 50, 80, 90, 95, 99])
    print('train q-error: 30%:', q_error[0], '  50%:', q_error[1], '  80%:', q_error[2], '  90%:', q_error[3], '  95%:', q_error[4], '  99%:', q_error[5])

print('done!')
torch.save(model.state_dict(), f'results/pretrain_params.pth')
print('model saved in results/pretrain_params.pth')
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
