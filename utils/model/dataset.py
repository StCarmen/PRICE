import os
import pickle
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_dataset_features(bin_size, dataset_list, train_or_test, usage):
    assert train_or_test in ['train', 'test'], f"train_or_test should be 'train' or 'test', but got {train_or_test}"
    assert usage in ['pretrain', 'finetune', 'test'], f"usage should be 'pretrain', 'finetune', or 'test', but got {usage}"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_features, true_cards, pg_est_cards = [], [], []
    n_join_cols, n_fanouts, n_tables, n_filter_cols = [], [], [], []
    train_len = 0 if train_or_test == 'train' else None
    test_list_lens = [0 for _ in range(len(dataset_list))] if train_or_test == 'test' else None

    for dataset in dataset_list:
        path = f"{current_dir}/../../setup/features/{usage}/{dataset}/features{bin_size}.pkl"
        print(f"using {dataset} path: {path}")

        with open(path, 'rb') as file:
            features = pickle.load(file)
        data_features.extend(features['data_features'])
        true_cards.extend(features['true_cards'])
        pg_est_cards.extend(features['pg_est_cards'])
        n_join_cols.extend(features['n_join_cols'])
        n_fanouts.extend(features['n_fanouts'])
        n_tables.extend(features['n_tables'])
        n_filter_cols.extend(features['n_filter_cols'])
        curr_len = len(features['true_cards'])
        
        if train_or_test == 'train':
            train_len += curr_len
        else:
            test_list_lens[dataset_list.index(dataset)] += curr_len
        print(f"{dataset} loading done!!, len: {curr_len}")
    
    if train_or_test == 'train':
        print(f"train_len: {train_len}")
        return data_features, true_cards, pg_est_cards, n_join_cols, n_fanouts, n_tables, n_filter_cols
    else:
        return data_features, true_cards, pg_est_cards, n_join_cols, n_fanouts, n_tables, n_filter_cols, test_list_lens

def make_feature_datasets(data_features, labels, pg_est_cards, padding_masks,
                          n_join_cols, n_fanouts, n_tables, n_filter_cols, 
                          train_or_test, test_list_lens=None):
    assert train_or_test in ['train', 'test'], f"train_or_test should be 'train' or 'test', but got {train_or_test}"
    assert test_list_lens is not None if train_or_test == 'test' else True, "test_list_lens should be provided for test dataset"

    if train_or_test == 'train':
        train_dataset = CustomDataset(list(zip(data_features, labels, pg_est_cards, padding_masks,
                                               n_join_cols, n_fanouts, n_tables, n_filter_cols)))
        return train_dataset
    else:
        test_datasets_list = []
        start_idx, end_idx = 0, 0
        for i, test_len in enumerate(test_list_lens):
            start_idx = start_idx if i == 0 else end_idx
            end_idx = end_idx + test_len

            test_dataset = CustomDataset(list(zip(data_features[start_idx:end_idx], labels[start_idx:end_idx],
                                                  pg_est_cards[start_idx:end_idx], padding_masks[start_idx:end_idx],
                                                  n_join_cols[start_idx:end_idx], n_fanouts[start_idx:end_idx],
                                                  n_tables[start_idx:end_idx], n_filter_cols[start_idx:end_idx])))
            test_datasets_list.append(test_dataset)
        return test_datasets_list
    
def make_train_feature_dataloaders(train_dataset, train_batch_size):
    return DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

def make_test_feature_dataloaders(test_datasets_list, test_list_lens):
    return [DataLoader(test_dataset, batch_size=test_list_lens[i], shuffle=False) for i, test_dataset in enumerate(test_datasets_list)]
