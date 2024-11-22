import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gpt1 import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from utils import (LeaveOneOutDataset, WarmUpScheduler,
                   calc_leave_one_out_full, calc_leave_one_out_partial,
                   calc_successive_metrics, data_to_sequences, get_dataloader)

torch.manual_seed(41)
np.random.seed(41)


@click.command()
@click.argument("model_name")
@click.option("--validate_batch_size", "-vbs", default=128)
@click.option("--full_eval", is_flag=True)
@click.option("--use_zvuk", is_flag=True)
def main(model_name, validate_batch_size, full_eval, use_zvuk):
    model = torch.load(f"models/{model_name}.pt")
    trajectory_len = model.block_size // 3
    # get and fix Danil get_dataset
    columns_mapping = {
        "userid": "user_idx",
        "itemid": "item_idx",
        "rating": "relevance",
        "timestamp": "timestamp",
    }
    inverse_columns_mapping = {value: key for key, value in columns_mapping.items()}

    def read_and_rename(path, use_csv=False):
        return (
            pd.read_csv(path).rename(columns=columns_mapping)
            if use_csv
            else pd.read_parquet(path).rename(columns=columns_mapping)
        )

    print("read data")
    if use_zvuk:
        # training_temp = read_and_rename("data_split/training_temp.parquet")
        testset_valid_temp = read_and_rename(
            "data_split/zvuk/testset_valid_temp.parquet"
        )
        testset = read_and_rename("data_split/zvuk/testset.parquet")
        holdout_valid_temp = read_and_rename(
            "data_split/zvuk/holdout_valid_temp.parquet"
        )
    else:
        # training_temp = read_and_rename("data_split/training_temp.csv", use_csv=True)
        testset_valid_temp = read_and_rename(
            "data_split/testset_valid_temp.csv", use_csv=True
        )
        testset = read_and_rename("data_split/testset.csv", use_csv=True)
        holdout_valid_temp = read_and_rename(
            "data_split/holdout_valid_temp.csv", use_csv=True
        )
    print("finish")
    #

    item_num = testset_valid_temp["item_idx"].max() + 1
    user_num = testset_valid_temp["user_idx"].max() + 1

    # create validate_datalaoder
    last_df = (
        testset_valid_temp.sort_values(["user_idx", "timestamp"])
        .groupby("user_idx")
        .tail(trajectory_len - 1)
    )
    validate_dataset = LeaveOneOutDataset(last_df, user_num, item_num, trajectory_len)
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=validate_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    #

    if full_eval:
        metrics = calc_leave_one_out_full(
            model, validate_dataloader, testset_valid_temp, holdout_valid_temp
        )
    else:
        metrics = calc_leave_one_out_partial(
            model, validate_dataloader, 30, testset_valid_temp, holdout_valid_temp
        )

    print(metrics)


if __name__ == "__main__":
    main()
