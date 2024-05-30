import argparse


def get_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--bin_size', type=int, default=40, help='')
    arg_parser.add_argument('--table_dim', type=int, default=4, help='')
    arg_parser.add_argument('--filter_dim', type=int, default=43, help='')

    arg_parser.add_argument('--query_hidden_dim', type=int, default=512, help='')
    arg_parser.add_argument('--final_hidden_dim', type=int, default=1024, help='')
    arg_parser.add_argument('--output_dim', type=int, default=1, help='')
    arg_parser.add_argument('--n_embd', type=int, default=256, help='')
    arg_parser.add_argument('--n_layers', type=int, default=6, help='')
    arg_parser.add_argument('--n_heads', type=int, default=8, help='')
    arg_parser.add_argument('--dropout_rate', type=float, default=0.2, help='')
    
    arg_parser.add_argument('--batch_size', type=int, default=15000, help='')
    arg_parser.add_argument('--lr', type=float, default=2.85e-5, help='')
    arg_parser.add_argument('--wd', type=float, default=5e-5, help='')
    arg_parser.add_argument('--step_size', type=int, default=5, help='')
    arg_parser.add_argument('--gamma', type=float, default=0.90, help='')

    args = arg_parser.parse_args()
    return args

def get_args_finetune():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--bin_size', type=int, default=40, help='')
    arg_parser.add_argument('--table_dim', type=int, default=4, help='')
    arg_parser.add_argument('--filter_dim', type=int, default=43, help='')

    arg_parser.add_argument('--query_hidden_dim', type=int, default=512, help='')
    arg_parser.add_argument('--final_hidden_dim', type=int, default=1024, help='')
    arg_parser.add_argument('--output_dim', type=int, default=1, help='')
    arg_parser.add_argument('--n_embd', type=int, default=256, help='')
    arg_parser.add_argument('--n_layers', type=int, default=6, help='')
    arg_parser.add_argument('--n_heads', type=int, default=8, help='')
    arg_parser.add_argument('--dropout_rate', type=float, default=0.2, help='')
    
    arg_parser.add_argument('--batch_size', type=int, default=15000, help='')
    arg_parser.add_argument('--lr', type=float, default=2.85e-5, help='')
    arg_parser.add_argument('--wd', type=float, default=5e-5, help='')
    arg_parser.add_argument('--step_size', type=int, default=5, help='')
    arg_parser.add_argument('--gamma', type=float, default=0.90, help='')

		arg_parser.add_argument('--dataset', type=str, default='genome', help='')

    args = arg_parser.parse_args()
    return args
