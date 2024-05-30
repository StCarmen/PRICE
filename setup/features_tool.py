import os
import torch    
import numpy as np
import sqlglot
import pickle


class Sql2Feature():
    def __init__(self, database, bin_size, usage):
        """ initialize Sql2Feature
        :param database: database name (type: str)
        :param bin_size: bin size (type: int)
        :param usage: pretrain, finetune, test (type: str)
        """
        super().__init__()
        self.database = database
        self.bin_size = bin_size
        self.usage = usage

        self.columns_distributions = {}
        self.columns_bin_edges = {}
        self.columns_summary = {}

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_hist = f'{current_dir}/../datas/statistics/{usage}/{database}/histogram{self.bin_size}.pkl'
        with open(file_hist, 'rb') as f:
            self.information_histogram = pickle.load(f)
        file_fanout = f'{current_dir}/../datas/statistics/{usage}/{database}/fanout{self.bin_size}.pkl'
        with open(file_fanout, 'rb') as f:
            self.information_fanout = pickle.load(f)
        file_summary = f'{current_dir}/../datas/statistics/{usage}/{database}/summary{self.bin_size}.pkl'
        with open(file_summary, 'rb') as f:
            self.information_summary = pickle.load(f)
        file_size = f'{current_dir}/../datas/statistics/{usage}/{database}/size.pkl'
        with open(file_size, 'rb') as f:
            self.information_size = pickle.load(f)
        file_col_type = f'{current_dir}/../datas/statistics/{usage}/{database}/abbrev_col_type.pkl'
        with open(file_col_type, 'rb') as f:
            self.information_coltype = pickle.load(f)

    def parse_sql(self, sql):
        """ parse sql statement
        :param sql: sql statement (type: str)
        :return: columns, tables, joins, ref_to_tables (type: list[str], list[str], list[str], dict[str, str])

        >>> parse_sql(sql):
        ['p.score', 'ph.creationdate', 'p.creationdate', 'p.id', 'pl.postid', 'ph.postid']
        ['p', 'pl', 'ph']
        ['p.id = pl.postid', 'pl.postid = ph.postid']
        {'p': 'posts', 'pl': 'postlinks', 'ph': 'posthistory'}
        """
        paresd_sql = sqlglot.parse_one(sql)

        columns = []    # get all columns in a sql statement
        for column in paresd_sql.find_all(sqlglot.exp.Column):
            if str(column) not in columns:
                columns.append(str(column))
        
        tables = []     # get all tables in a sql statement
        ref_to_tables = {}  # transform table alias name -> table name
        for table in paresd_sql.find_all(sqlglot.exp.Table):
            if table.alias_or_name not in tables:
                tables.append(table.alias_or_name)
                ref_to_tables[table.alias_or_name] = table.name
        
        joins = []      # get all join tabls in a sql statement
        for eq in paresd_sql.args["where"].find_all(sqlglot.exp.EQ):
            if isinstance(eq.args["expression"], sqlglot.exp.Column):
                joins.append(str(eq))
        return columns, tables, joins, ref_to_tables
    
    def get_summary_location(self, sql, column):
        """ get summary location used for calculating selectivity
        """
        parsed_sql = sqlglot.parse_one(sql)
        filter_number = None
        for filters in parsed_sql.args["where"].find_all(sqlglot.exp.EQ):
            if column != str(filters.args["this"]):     # if not the column we want
                continue
            if isinstance(filters.args["expression"], sqlglot.exp.Column):  # pass join type EQ
                continue
            
            try:
                filter_number = int(str(filters.args["expression"]))
            except ValueError:
                try:
                    filter_number = float(str(filters.args["expression"]))
                except ValueError:
                    filter_number = str(filters.args["expression"]).replace("'", "")

        assert filter_number is not None
        return filter_number
    
    def calculate_hist_selectivity(self, histogram, bin_edges, range_low, range_high):
        """ calculate selectivity based on histogram, we use accumulate distribution / total distribution

        :param histogram: histogram (type: list[int])
        :param bin_edges: bin edges (type: list[float])
        :param range_low: range's low bound (type: float)
        :param range_high: range's high bound (type: float)
        :return: selectivity (type: float)
        """
        begin_sum = 0
        end_sum = 0

        # accumulate distribution
        for j, bin_edge in enumerate(bin_edges):
            if bin_edge > range_low:
                begin_sum += histogram[j - 1] * (range_low - bin_edges[j - 1]) / (bin_edge - bin_edges[j - 1]) if j > 0 else 0
                break
            begin_sum += histogram[j - 1] if j > 0 else 0

        for j, bin_edge in enumerate(bin_edges):
            if bin_edge > range_high:
                end_sum += histogram[j - 1] * (range_high - bin_edges[j - 1]) / (bin_edge - bin_edges[j - 1]) if j > 0 else 0
                break
            end_sum += histogram[j - 1] if j > 0 else 0
        return end_sum - begin_sum    

    def flatten_list(self, nested_list):
        """ flatten a nested list

        :param nested_list: nested list (type: list)
        :return: flattened list (type: list)

        >>> flatten_list([1, [2], 3, [4, 5, 6], [7, 8], 9])
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        """
        flattened_list = []
        for item in nested_list:
            if isinstance(item, list):
                flattened_list.extend(self.flatten_list(item))
            else:
                flattened_list.append(item)
        return flattened_list
    
    def get_column_histograms(self, column):
        """ get column histograms

        :param column: column name (type: str)
        :return: column histograms features (type: list[float])
        """
        column_t, column_c = column.split('.')[0], column.split('.')[1]
        hist, bin_edges = self.information_histogram[column_t][column_c]['hist'], self.information_histogram[column_t][column_c]['bin_edges']
        self.columns_distributions[column] = hist
        self.columns_bin_edges[column] = bin_edges

        return list(self.columns_distributions[column] / self.information_histogram[column_t][column_c]['len'])
    
    def get_fanout_features(self, join):
        """ get fanout features

        :param join: join condition (type: str)
        :return: left_fanout, right_fanout (type: list[float], list[float])
        """
        left_join, right_join = join.split(" = ")[0], join.split(" = ")[1]
        key = (left_join, right_join)
        return self.information_fanout[key][0], self.information_fanout[key][1]
    
    def calculate_summary_selectivity(self, keys, values, location):
        """ return the selectivity of the location in the summary
        """
        try:
            idx = keys.index(location)
            return values[idx]
        except ValueError:
            return values[-1]
    
    def space_saving_summary(self, column):
        """ space saving summary algorithm

        :param column: column name (type: str)
        :return: summary (type: list[int])
        """
        k = self.bin_size - 1

        if column not in self.columns_summary.keys():
            table_alias_name, _ = column.split(".")[0], column.split(".")[1]
            column_t, column_c = column.split('.')[0], column.split('.')[1]

            summary = self.information_summary[column_t][column_c]

            if len(summary['keys']) < self.bin_size:        
                keys, values = summary['keys'], summary['values']
                padding_keys = [-1e3] * (k - len(keys) + 1) if len(keys) < k else [-1e3]
                padding_values = [0] * (k - len(values) + 1) if len(values) < k else [0]
                keys, values = keys + padding_keys, values + padding_values
                assert len(keys) == k + 1 and len(values) == k + 1, f"keys and values should be equal to {k + 1}, but get {len(keys)} and {len(values)}"
            else:
                # get top k keys and values and others are merged into the last one
                keys, values = summary['keys'], summary['values']
                # sort via values: from large to small
                values, keys = zip(*sorted(zip(values, keys), reverse=True))
                # TODO: top 40, not the same as before
                keys = list(keys[:k]) + ['OtHeRs']
                values = list(values[:k]) + [sum(values[k:]) / (len(values) - k)]
                assert len(keys) == k + 1 and len(values) == k + 1, f"keys and values should be equal to {k + 1}, but get {len(keys)} and {len(values)}"
            self.columns_summary[column] = keys, values
        return self.columns_summary[column]
    
    def get_summary_ranges(self, sql, column, keys):
        """ get summary ranges like what we do in the histogram, ranges are used in model learning

        :param sql: sql statement (type: str)
        :param column: column name (type: str)
        :param keys: summary keys (type: list[str, int])
        :return: summary ranges (type: list[float])
        """
        parsed_sql = sqlglot.parse_one(sql)
        for filters in parsed_sql.args["where"].find_all(sqlglot.exp.EQ):
            if column != str(filters.args["this"]):     # if not the column we want
                continue
            if isinstance(filters.args["expression"], sqlglot.exp.Column):  # pass join type EQ
                continue
            
            # summary key: int, float, str
            try:
                filter = int(str(filters.args["expression"]))
            except ValueError:
                try:
                    filter = float(str(filters.args["expression"]))
                except ValueError:
                    filter = str(filters.args["expression"]).replace("'", "")
            try:
                idx = keys.index(filter)
                return [idx / self.bin_size, (idx + 1) / self.bin_size]
            except ValueError:
                return [((self.bin_size - 1) / self.bin_size), 1.0]
        
        return None

    def get_table_size(self, table):
        """ get table size features

        :param table: table name (type: str)
        :return: table size (type: int)
        """
        return self.information_size[table]['size']
    
    def get_filter_norm_range(self, sql, column, bin_edges):
        """ get filter range bounds normalized by min and max values, ranges are used in model learning

        :param sql: sql statement (type: str)
        :param column: column name (type: str)
        :param bin_edges: bin edges (type: list[float])
        :return: normalized range's low bound, normalized range's high bound (type: float, float)
        """
        bin_min, bin_max = 0.0, 1.0
        filter_types = [sqlglot.exp.EQ, sqlglot.exp.NEQ, sqlglot.exp.LTE, sqlglot.exp.LT, sqlglot.exp.GTE, sqlglot.exp.GT]
        parsed_sql = sqlglot.parse_one(sql)

        for filter_type in filter_types:
            # find the specific filter type like <=, <
            for filters in parsed_sql.args["where"].find_all(filter_type):
                if column != str(filters.args["this"]):     # if not the column we want
                    continue
                if isinstance(filters.args["expression"], sqlglot.exp.Column):  # pass join type EQ
                    continue
                
                filter_number = float(str(filters.args["expression"]))

                if filter_type == filter_types[0]:  # EQ
                    bin_min = (filter_number - bin_edges[0]) / (bin_edges[-1] - bin_edges[0])
                    bin_max = (filter_number + 1e-5 - bin_edges[0]) / (bin_edges[-1] - bin_edges[0])
                    break
                if filter_type == filter_types[2]:  # LTE
                    bin_max = (filter_number + 1e-5 - bin_edges[0]) / (bin_edges[-1] - bin_edges[0])
                    break
                if filter_type == filter_types[3]:  # LT
                    bin_max = (filter_number - bin_edges[0]) / (bin_edges[-1] - bin_edges[0])
                    break
                if filter_type == filter_types[4]:  # GTE
                    bin_min = (filter_number - bin_edges[0]) / (bin_edges[-1] - bin_edges[0])
                    break
                if filter_type == filter_types[5]:  # GT
                    bin_min = (filter_number + 1e-5 - bin_edges[0]) / (bin_edges[-1] - bin_edges[0])
                    break
        # avoid out of range
        bin_min = 0.0 if bin_min < 0.0 else bin_min
        bin_max = 1.0 if bin_max > 1.0 else bin_max
        
        assert bin_min >= 0.0 and bin_min <= 1.0 and bin_max >= 0.0 and bin_max <= 1.0 and bin_min <= bin_max

        return [bin_min, bin_max]
    
    def get_filter_ranges(self, sql, column):
        """ get filter column range bounds, ranges are used in calculating selectivity

        :param sql: sql statement (type: str)
        :param column: column name (type: str)
        :return: range's low bound, range's high bound (type: float, float)
        """
        bin_edges = self.columns_bin_edges[column]
        range_low, range_high = bin_edges[0], bin_edges[-1]
        filter_types = [sqlglot.exp.EQ, sqlglot.exp.NEQ, sqlglot.exp.LTE, sqlglot.exp.LT, sqlglot.exp.GTE, sqlglot.exp.GT]
        parsed_sql = sqlglot.parse_one(sql)

        for filter_type in filter_types:
            # find the specific filter type like <=, <
            for filters in parsed_sql.args["where"].find_all(filter_type):
                if column != str(filters.args["this"]):     # if not the column we want
                    continue
                if isinstance(filters.args["expression"], sqlglot.exp.Column):  # pass join type EQ
                    continue
                
                # histogram: int, float
                try:
                    filter_number = int(str(filters.args["expression"]))
                except ValueError:
                    filter_number = float(str(filters.args["expression"]))

                if filter_type == filter_types[0]:  # EQ
                    range_low = filter_number
                    range_high = filter_number + 1e-5
                    break
                if filter_type == filter_types[2]:  # LTE
                    range_high = filter_number + 1e-5
                    break
                if filter_type == filter_types[3]:  # LT
                    range_high = filter_number
                    break
                if filter_type == filter_types[4]:  # GTE
                    range_low = filter_number
                    break
                if filter_type == filter_types[5]:  # GT
                    range_low = filter_number + 1e-5
                    break
        
        return range_low, range_high

    def create_sql_features(self, sql):
        """ create sql features

        :param sql: sql statement (type: str)
        :return: sql_features, n_join_col, n_fanout, n_table, n_filter_col (type: tuple)
        """
        
        columns, tables, joins, ref_to_tables = self.parse_sql(sql)
        if len(tables) != len(joins) + 1:
            print(f"error: {sql}")
            return None
        assert len(tables) == len(joins) + 1, "table number should be equal to join number + 1"

        table_join_cols_dict, table_filter_cols_dict = {}, {}
        for table in tables:
            table_join_cols_dict[table], table_filter_cols_dict[table] = [], []

        # NOTE: There is some problem when a column appears 
        # in join condition and filter condition at the same time
        for column in columns:
            table, _ = column.split(".")[0], column.split(".")[1]
            join_flag = False
            for join in joins:
                left_join_col, right_join_col = join.split("=")[0].strip(), join.split("=")[1].strip()
                if column == left_join_col or column == right_join_col:
                    join_flag = True
                    break
            if join_flag:
                table_join_cols_dict[table].append(column)
            else:
                table_filter_cols_dict[table].append(column)
        
        join_columns = self.flatten_list(list(table_join_cols_dict.values()))
        filter_columns = self.flatten_list(list(table_filter_cols_dict.values()))

        table_features, table_sels = [], {}
        for table in tables:
            table_sels[table] = []

        join_column_histograms = []
        for join_column in join_columns:
            join_column_histogram = self.get_column_histograms(join_column)
            join_column_histograms.append(torch.tensor(join_column_histogram))

        # NOTE: filter column maybe none
        filter_column_features = []
        for filter_column in filter_columns:
            if filter_column.split('.')[-1] in self.information_coltype['col_type'][filter_column.split('.')[0]]['dsct']:
                keys, values = self.space_saving_summary(filter_column)
                filter_column_histogram = torch.tensor(values) / self.get_table_size(filter_column.split(".")[0])
                summary = self.get_summary_ranges(sql, filter_column, keys)
                if summary is not None:
                    filter_column_ranges = torch.tensor(summary)
                    location = self.get_summary_location(sql, filter_column)
                    filter_column_selectivity = torch.tensor([self.calculate_summary_selectivity(keys, values, location) / self.get_table_size(filter_column.split(".")[0])])
                    assert filter_column_selectivity.item() != 0.0, f"selectivity should not be 0, but get {filter_column_selectivity.item()}"
                    table_sels[filter_column.split(".")[0]].append(filter_column_selectivity.item())
                else:  # if a dsct column without '=' condition, but use other conditions like '<', '>'
                    filter_column_histogram = torch.tensor(self.get_column_histograms(filter_column))
                    filter_column_ranges = torch.tensor(self.get_filter_norm_range(sql, filter_column, self.columns_bin_edges[filter_column]))
                    range_low, range_high = self.get_filter_ranges(sql, filter_column)
                    distribution, bin_edges = self.columns_distributions[filter_column], self.columns_bin_edges[filter_column]
                    filter_column_selectivity = torch.tensor([self.calculate_hist_selectivity(distribution, bin_edges, range_low, range_high) / self.get_table_size(filter_column.split(".")[0])])
                    assert filter_column_selectivity.item() != 0.0, f"selectivity should not be 0, but get {filter_column_selectivity.item()}"
                    table_sels[filter_column.split(".")[0]].append(filter_column_selectivity.item())
            else:
                filter_column_histogram = torch.tensor(self.get_column_histograms(filter_column))
                filter_column_ranges = torch.tensor(self.get_filter_norm_range(sql, filter_column, self.columns_bin_edges[filter_column]))
                range_low, range_high = self.get_filter_ranges(sql, filter_column)
                distribution, bin_edges = self.columns_distributions[filter_column], self.columns_bin_edges[filter_column]
                filter_column_selectivity = torch.tensor([self.calculate_hist_selectivity(distribution, bin_edges, range_low, range_high) / self.get_table_size(filter_column.split(".")[0])])
                assert filter_column_selectivity.item() != 0.0, f"selectivity should not be 0, but get {filter_column_selectivity.item()}"
                table_sels[filter_column.split(".")[0]].append(filter_column_selectivity.item())

            filter_column_features.append(torch.cat([filter_column_histogram, filter_column_ranges, filter_column_selectivity]))
        
        # avi, minsel, ebo
        for table in tables:
            if len(table_sels[table]) == 0:
                avi, minsel, ebo = torch.tensor([1]), torch.tensor([1]), torch.tensor([1])
            else:
                avi = torch.prod(torch.tensor(table_sels[table]))
                minsel = torch.min(torch.tensor(table_sels[table]))
                sorted_sels = sorted(table_sels[table], reverse=True)
                ebo = 1
                for i in range(len(sorted_sels)):
                    if i > 3:
                        break
                    ebo = ebo * sorted_sels[i] ** (1 / (2 ** i))
                ebo = torch.tensor([ebo])
            table_size = self.get_table_size(table)

            # we assume that the filtering range will not exceed the maximum value 
            # or be less than the minimum value
            assert avi.item() != 0.0 and minsel.item() != 0.0 and ebo.item() != 0.0, f"avi, minsel, ebo should not be 0, but get {avi.item()}, {minsel.item()}, {ebo.item()}"
            table_features.append(torch.cat([torch.tensor([np.log(table_size)]), torch.tensor([avi]), torch.tensor([minsel]), torch.tensor([ebo])]))
        
        fanout_features = []
        # there may be a join column appearing multiple times in the join condition, 
        # but fanout will be different, so we need to calculate the fanout of each join condition
        for join in joins:
            fanout_features1, fanout_features2 = self.get_fanout_features(join)
            fanout_features.append(torch.tensor(fanout_features1))
            fanout_features.append(torch.tensor(fanout_features2))

        tensor_list = [join_column_histograms, fanout_features, table_features, filter_column_features]
        tensor_list = self.flatten_list(tensor_list)
        sql_feature_len = len(torch.cat(tensor_list))
        
        sql_features = torch.cat(join_column_histograms), torch.cat(fanout_features), \
            torch.cat(table_features), torch.cat(filter_column_features) if len(filter_column_features) > 0 else None
        
        length = self.bin_size * (len(join_columns) + len(joins) * 2) + 4 * len(tables) + (self.bin_size + 3) * len(filter_columns)
        assert sql_feature_len == length, f"{self.bin_size} dataset length should be {length}, but get {sql_feature_len}"

        assert len(join_columns) >= len(tables), f"join column number should be greater than table number, but get {len(join_columns)} and {len(tables)}"
        assert len(joins) == len(tables) - 1, f"join number should be equal to table number - 1, but get {len(joins)} and {len(tables)}"

        # dataset contains 4 type features
        return sql_features, len(join_columns), len(joins) * 2, len(tables), len(filter_columns)
