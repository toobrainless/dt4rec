"""
This script is used for dataset preparation, including preprocessing, splitting the dataset into subsets, 
and saving the resulting train and test sets. It also generates sequential data suitable for model training.
"""

import pandas as pd
import torch

from preprocessing import preprocessing
from utils import MyIndexer, get_all_seqs, split_last_n

df = pd.read_parquet("zvuk-interactions.parquet")
df["timestamp"] = df["datetime"].astype("int64")
print(df.columns)
filtered_data = preprocessing(
    df,
    users_sample=None,
    item_min_count=101,
    min_len=10,
    core=False,
    encoding=False,
    drop_repeats=False,
    user_id="user_id",
    item_id="track_id",
    timestamp="timestamp",
    path_to_save=None,
)
filtered_data = filtered_data.rename(
    columns={"user_id": "user_idx", "item_id": "item_idx"}
)
filtered_data = filtered_data.sort_values("timestamp").reset_index(drop=True)

train, test = split_last_n(filtered_data, "user_idx", "item_idx")
indexer = MyIndexer(user_col="user_idx", item_col="item_idx")
train = indexer.fit_transform(train).reset_index(drop=True)
test = indexer.transform(test).reset_index(drop=True)

print("train", train.user_idx.nunique(), train.user_idx.max(), train.user_idx.min())
print(
    "test", test.user_idx.nunique(), test.user_idx.max(), test.user_idx.min(), len(test)
)

train.to_parquet("train.parquet")
test.to_parquet("test.parquet")

all_seqs = get_all_seqs(train, 100, train.item_idx.max(), 0)
torch.save(all_seqs, "all_seqs.pt")
