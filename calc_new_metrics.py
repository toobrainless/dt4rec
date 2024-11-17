import pandas as pd
import torch

from utils import calc_successive_metrics, data_to_sequences

device = torch.device("cuda")


columns_mapping = {
    "userid": "user_idx",
    "itemid": "item_idx",
    "rating": "relevance",
    "timestamp": "timestamp",
}
inverse_columns_mapping = {value: key for key, value in columns_mapping.items()}


def read_and_rename(path):
    return pd.read_csv(path).rename(columns=columns_mapping)


testset = read_and_rename("data_split/testset.csv")

data_description_temp = {
    "users": "userid",
    "items": "itemid",
    "order": "timestamp",
    "n_users": 5400,
    "n_items": 3658,
}

test_sequences = data_to_sequences(
    testset.rename(columns=inverse_columns_mapping), data_description_temp
)

for model_name in [
    "emb_64_traj_100_default",
    "emb_64_traj_100_svd_freeze",
    "emb_64_traj_100_svd_unfreeze",
]:
    print(model_name)
    model = torch.load(f"/home/hdilab/amgimranov/dt4rec/models/{model_name}.pt")
    metrics = calc_successive_metrics(model, test_sequences, data_description_temp)
    print(metrics)
    print()
