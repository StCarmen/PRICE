import torch


def features_padding(bin_size, table_dim, filter_dim, dataset, n_join_cols, n_fanouts, n_tables, n_filter_cols, 
                     max_n_join_col=None, max_n_fanout=None, max_n_table=None, max_n_filter_col=None):
    """ padding dataset to the same length

    :param dataset: dataset to be padded (type: list[torch.tensor])
    :param n_join_cols: number of join histograms of dataset (type: list[int])
    :param n_fanouts: number of fanout features of dataset (type: list[int])
    :param n_tables: number of table_sizes of dataset (type: list[int])
    :param n_filter_cols: number of filter_infos of dataset (type: list[int])
    :param max_n_join_col: if model is pretrained on max_n_join_col, then max_n_join_col should be set as before (type: int)
    :param max_n_fanout: if model is pretrained on max_n_fanout, then max_n_fanout should be set as before (type: int)
    :param max_n_table: if model is pretrained on max_n_table, then max_n_table should be set as before (type: int)
    :param max_n_filter_col: if model is pretrained on max_n_filter_col, then max_n_filter_col should be set as before (type: int)
    :return: padded dataset (type: list[torch.tensor])
    """
    max_n_filter_col = max(n_filter_cols) if max_n_filter_col is None else max_n_filter_col
    max_n_join_col = max(n_join_cols) if max_n_join_col is None else max_n_join_col
    max_n_fanout = max(n_fanouts) if max_n_fanout is None else max_n_fanout
    max_n_table = max(n_tables) if max_n_table is None else max_n_table

    max_n_feature = max_n_join_col + max_n_fanout + max_n_table + max_n_filter_col
    print(f"max_n_join_col is {max_n_join_col}, max_n_fanout is {max_n_fanout}, max_n_table is {max_n_table}, max_n_filter_col is {max_n_filter_col}")
    print(f"max_n_feature is {max_n_feature}!")
    length = bin_size * (max_n_join_col + max_n_fanout) + table_dim * max_n_table + filter_dim * max_n_filter_col
    
    padding_masks = []
    for i, (n_join_col, n_fanout, n_table, n_filter_col) in enumerate(list(zip(n_join_cols, n_fanouts, n_tables, n_filter_cols))):
        padding_mask = [1] + [1] * n_join_col + [0] * (max_n_join_col - n_join_col) + \
                    [1] * n_fanout + [0] * (max_n_fanout - n_fanout) + \
                    [1] * n_table + [0] * (max_n_table - n_table) + \
                    [1] * n_filter_col + [0] * (max_n_filter_col - n_filter_col)
        assert len(padding_mask) == max_n_feature + 1, f"len(padding_mask) should be {max_n_feature + 1}, but got {len(padding_mask)}"
        padding_masks.append(torch.tensor(padding_mask))
        
        join_column_histograms, fanout_features, table_features, filter_column_features = dataset[i]
        
        assert n_join_col <= max_n_join_col and n_fanout <= max_n_fanout and \
            n_table <= max_n_table and n_filter_col <= max_n_filter_col, \
                "n_feature should be less than max_n_feature"

        padding_value = -1e3
        if n_join_col < max_n_join_col:
            padding_n_join_col = max_n_join_col - n_join_col
            padding_tensor = torch.tensor([padding_value] * bin_size)
            for _ in range(int(padding_n_join_col)):
                # padding histogram in the end
                join_column_histograms = torch.cat([join_column_histograms, padding_tensor])
        if n_fanout < max_n_fanout:
            padding_n_fanout = max_n_fanout - n_fanout
            padding_tensor = torch.tensor([padding_value] * bin_size)
            for _ in range(int(padding_n_fanout)):
                # padding fanout in the end
                fanout_features = torch.cat([fanout_features, padding_tensor])
        if n_table < max_n_table:
            padding_n_table = max_n_table - n_table
            padding_tensor = torch.tensor([0] + [padding_value] * (table_dim - 1))
            for _ in range(int(padding_n_table)):
                # padding table size in the end
                table_features = torch.cat([table_features, padding_tensor])
        if n_filter_col < max_n_filter_col:
            padding_n_filter_col = max_n_filter_col - n_filter_col
            # filter_column_histogram + filter_range + selectivity
            padding_tensor = torch.tensor([padding_value] * filter_dim)
            for _ in range(int(padding_n_filter_col)):
                # padding info in the end
                if filter_column_features is not None:
                    filter_column_features = torch.cat([filter_column_features, padding_tensor])
                else:
                    filter_column_features = padding_tensor
        
        dataset[i] = torch.cat([join_column_histograms, fanout_features, table_features, filter_column_features])
        assert len(dataset[i]) == length, f"len(dataset[{i}]) should be {length}, but got {len(dataset[i])}"

    return dataset, padding_masks
