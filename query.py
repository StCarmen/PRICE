import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from model.encoder import RegressionModel
from setup.features_tool import Sql2Feature
from utils.model.padding import features_padding


# These must match the configuration used to train the saved model.
BIN_SIZE = 40
TABLE_DIM = 4
FILTER_DIM = 43
QUERY_HIDDEN_DIM = 512
FINAL_HIDDEN_DIM = 1024
OUTPUT_DIM = 1
N_EMBD = 256
N_LAYERS = 6
N_HEADS = 8
DROPOUT_RATE = 0.2

# These are the max feature counts printed during evaluation for this model.
# If you retrain with different settings, update them accordingly.
MAX_N_JOIN_COL = 9
MAX_N_FANOUT = 12
MAX_N_TABLE = 7
MAX_N_FILTER_COL = 10


def lower_except_quotes(s: str) -> str:
    """Lowercase SQL except inside quotes (matches features_generate behavior)."""
    inside_quote = False
    quote_char = ""
    result = []

    for char in s:
        if char in "'\"" and (not inside_quote or quote_char == char):
            inside_quote = not inside_quote
            quote_char = "" if inside_quote is False else char
        if not inside_quote:
            result.append(char.lower())
        else:
            result.append(char)

    return "".join(result)


def load_trained_model(device: torch.device) -> RegressionModel:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{current_dir}/results/model_params.pth"
    print(f"load model from {model_path}")

    # Build model with the same architecture used during training
    model = RegressionModel(
        n_join_col=MAX_N_JOIN_COL,
        n_fanout=MAX_N_FANOUT,
        n_table=MAX_N_TABLE,
        n_filter_col=MAX_N_FILTER_COL,
        hist_dim=BIN_SIZE,
        table_dim=TABLE_DIM,
        filter_dim=FILTER_DIM,
        query_hidden_dim=QUERY_HIDDEN_DIM,
        final_hidden_dim=FINAL_HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_embd=N_EMBD,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    checkpoint = torch.load(model_path, map_location="cpu")

    # Our checkpoint is a raw state_dict with keys like "module.*"
    if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        state_dict = checkpoint
    else:
        state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip leading "module." when loading into a non-DataParallel model
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module.") :]] = v
        else:
            new_state_dict[k] = v

    incompat_keys = model.load_state_dict(new_state_dict, strict=False)
    missing, unexpected = incompat_keys.missing_keys, incompat_keys.unexpected_keys
    if missing:
        print("Warning: missing keys when loading state_dict:")
        for k in missing:
            print("  ", k)
    if unexpected:
        print("Warning: unexpected keys when loading state_dict:")
        for k in unexpected:
            print("  ", k)

    model.eval()
    return model


def predict_cardinality(sql: str, pg_est_card: float, database: str, usage: str = "test") -> float:
    """
    Convert a single SQL query to features and run the trained model to get
    a predicted cardinality.

    :param sql: SQL text (string)
    :param pg_est_card: PostgreSQL estimated cardinality for this query
    :param database: database name (e.g., imdb, stats, ergastf1, genome)
    :param usage: statistics usage partition (pretrain, finetune, test)
    :return: predicted cardinality (float)
    """
    device = torch.device("cpu")

    # Build feature extractor (uses precomputed statistics under datas/statistics/{usage}/{database})
    sql2feature = Sql2Feature(database=database, bin_size=BIN_SIZE, usage=usage)

    # Create model
    model = load_trained_model(device)

    # Feature extraction for a single query
    lowered_sql = lower_except_quotes(sql)
    ret = sql2feature.create_sql_features(lowered_sql)
    if ret is None:
        raise ValueError("Failed to create features for the given SQL (likely parse/join mismatch).")

    sql_features, n_join_col, n_fanout, n_table, n_filter_col = ret

    # Wrap as a "dataset" of size 1 and pad to the trained max sizes
    dataset = [sql_features]
    n_join_cols = [n_join_col]
    n_fanouts = [n_fanout]
    n_tables = [n_table]
    n_filter_cols = [n_filter_col]

    dataset, padding_masks = features_padding(
        BIN_SIZE,
        TABLE_DIM,
        FILTER_DIM,
        dataset,
        n_join_cols,
        n_fanouts,
        n_tables,
        n_filter_cols,
        max_n_join_col=MAX_N_JOIN_COL,
        max_n_fanout=MAX_N_FANOUT,
        max_n_table=MAX_N_TABLE,
        max_n_filter_col=MAX_N_FILTER_COL,
    )

    # Prepare tensors
    data = dataset[0].to(torch.float).to(device).unsqueeze(0)  # shape (1, feature_dim)
    padding_mask = padding_masks[0].to(device).unsqueeze(0)    # shape (1, max_n_feature+1)
    n_join_col_t = torch.tensor([[float(n_join_col)]], dtype=torch.float, device=device)
    n_fanout_t = torch.tensor([[float(n_fanout)]], dtype=torch.float, device=device)
    n_table_t = torch.tensor([[float(n_table)]], dtype=torch.float, device=device)
    n_filter_col_t = torch.tensor([[float(n_filter_col)]], dtype=torch.float, device=device)

    # Same scaling as in evaluate.py
    pg_est_card_t = torch.tensor([[float(pg_est_card)]], dtype=torch.float, device=device)
    pg_est_card_t = torch.log(pg_est_card_t + 1) + 1

    with torch.no_grad():
        output = model(
            data,
            pg_est_card_t,
            padding_mask,
            n_join_col_t,
            n_fanout_t,
            n_table_t,
            n_filter_col_t,
        ).view(1, -1)

    # Invert the scaling used in training: y = log(card + 1) + 1  =>  card = exp(y - 1) - 1
    y = output.cpu().numpy().astype(np.float64)
    card = np.exp(y - 1.0) - 1.0
    return float(card[0, 0])


def main():
    parser = argparse.ArgumentParser(description="Run a single SQL query through the trained PRICE model.")
    parser.add_argument("--sql", type=str, required=True, help="SQL query text.")
    parser.add_argument("--db", type=str, required=True, help="Database name (e.g., imdb, stats, ergastf1, genome).")
    parser.add_argument(
        "--pg_est",
        type=float,
        required=True,
        help="PostgreSQL estimated cardinality for this query (the model learns to correct this).",
    )
    parser.add_argument(
        "--usage",
        type=str,
        default="test",
        choices=["pretrain", "finetune", "test"],
        help="Which statistics partition to use (default: test).",
    )

    args = parser.parse_args()

    pred_card = predict_cardinality(args.sql, args.pg_est, args.db, args.usage)
    print(f"SQL: {args.sql}")
    print(f"PG estimate: {args.pg_est}")
    print(f"Predicted cardinality: {pred_card}")


if __name__ == "__main__":
    main()
