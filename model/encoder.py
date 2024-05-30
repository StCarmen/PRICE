import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module import ScaleEmbedding, FilterEmbedding, Encoder


class RegressionModel(nn.Module):
    def __init__(self, n_join_col, n_fanout, n_table, n_filter_col,
                 hist_dim, table_dim, filter_dim,
                 query_hidden_dim, final_hidden_dim, output_dim,
                 n_embd, n_layers, n_heads, dropout_rate):
        super(RegressionModel, self).__init__()
        self.n_join_col, self.n_fanout, self.n_table, self.n_filter_col = n_join_col, n_fanout, n_table, n_filter_col
        print(f"n_features: {n_join_col + n_fanout + n_table + n_filter_col}!")
        self.hist_dim, self.table_dim, self.filter_dim = hist_dim, table_dim, filter_dim
        self.query_hidden_dim, self.final_hidden_dim, self.output_dim = query_hidden_dim, final_hidden_dim, output_dim
        self.n_embd, self.n_layers, self.n_heads = n_embd, n_layers, n_heads
        self.dropout_rate = dropout_rate

        self.scale_embedding = ScaleEmbedding(n_join_col, n_fanout, hist_dim, n_embd)
        self.filter_embedding = FilterEmbedding(n_join_col, n_fanout, n_table, n_filter_col,
                                          hist_dim, table_dim, filter_dim, n_embd)
        self.scale_encoder = Encoder(n_embd, n_layers, n_heads, dropout_rate)
        self.filter_encoder = Encoder(n_embd, n_layers, n_heads, dropout_rate)

        self.len_net = nn.Linear(4, 16)
        self.linear = nn.Linear(n_embd + 16, query_hidden_dim)
        self.elu = nn.ELU()
        self.output = nn.Linear(query_hidden_dim, output_dim)

        self.final_linear1 = nn.Linear(3, final_hidden_dim)
        self.final_linear2 = nn.Linear(final_hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x, pg_est_card=None, padding_mask=None, n_join_col=None, n_fanout=None, n_table=None, n_filter_col=None):
        # scaling stage
        scale_features = self.scale_embedding(x)
        masks1 = padding_mask[:, :1 + self.n_join_col + self.n_fanout] if padding_mask is not None else None
        scaling_output = self.scale_encoder(scale_features, masks1)
        
        # filtering stage
        filter_features = self.filter_embedding(scaling_output, x)
        masks2 = padding_mask[:, :] if padding_mask is not None else None
        filtering_output = self.filter_encoder(filter_features, masks2)
        query_output = filtering_output[:, 0, :]

        len_features = torch.cat([n_join_col, n_fanout, n_table, n_filter_col], dim=1)
        len_features = self.len_net(len_features)

        query_output = self.linear(torch.cat([query_output, len_features], dim=1))
        query_output = self.elu(query_output)
        query_output = F.dropout(query_output, p=self.dropout_rate, training=self.training)
        output = self.output(query_output)

        table_sizes = []
        bias = self.n_join_col * self.hist_dim + self.n_fanout * self.hist_dim
        for i in range(self.n_table):
            begin, end = bias + i * self.table_dim, bias + (i + 1) * self.table_dim
            table_size = x[:, begin:begin+1] # table_size, without avi, minsel, ebo
            table_sizes.append(table_size)
        cartesian_product = torch.sum(torch.stack(table_sizes, dim=1), dim=1)

        output = self.final_linear1(torch.cat([output, pg_est_card, cartesian_product], dim=1))
        output = self.relu(output)
        output = F.dropout(output, p=self.dropout_rate, training=self.training)
        output = self.final_linear2(output)
        return output
