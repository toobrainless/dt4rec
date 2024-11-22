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
                   calc_successive_metrics, data_to_sequences, get_dataloader)

torch.manual_seed(41)
np.random.seed(41)


@click.command()
@click.option("--train_batch_size", "-tbs", default=128)
@click.option("--validate_batch_size", "-vbs", default=128)
@click.option("--use_svd", default=False)
@click.option("--learn_svd", default=False)
@click.option("--trajectory_len", "-tl", default=100)
@click.option("--calc_successive", "-cs", default=False)
@click.option("--use_zvuk", "-uz", default=False)
@click.option("--full_eval", "-fe", False)
def main(
    train_batch_size,
    validate_batch_size,
    use_svd,
    learn_svd,
    trajectory_len,
    calc_successive,
    use_zvuk,
    full_eval,
):
    exp_name = f"use_zvuk_{use_zvuk}__trajectory_len_{trajectory_len}"
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

    # create model
    mconf = GPTConfig(
        user_num=user_num,
        item_num=item_num,
        vocab_size=item_num + 1,
        block_size=trajectory_len * 3,
        max_timestep=item_num,
    )
    model = GPT(mconf)

    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    print(f"Param num: {total_params}")

    if use_svd:
        item_embs = np.load("/home/hdilab/amgimranov/dt4rec/item_embs_ilya.npy")
        model.state_repr.item_embeddings.weight.data = torch.from_numpy(item_embs)
        model.state_repr.item_embeddings.weight.requires_grad = learn_svd
    #

    # create trainer
    tconf = TrainerConfig(epochs=1)

    train_dataloader = get_dataloader(
        testset_valid_temp,
        memory_size=3,
        seq_len=trajectory_len,
        pad_value=item_num,
        user_num=user_num,
        item_num=item_num,
        batch_size=train_batch_size,
    )

    optimizer = torch.optim.AdamW(
        model.configure_optimizers(),
        lr=3e-4,
        betas=(0.9, 0.95),
    )
    lr_scheduler = WarmUpScheduler(optimizer, dim_embed=768, warmup_steps=4000)

    tconf.update(optimizer=optimizer, lr_scheduler=lr_scheduler)
    trainer = Trainer(
        model,
        train_dataloader,
        tconf,
        exp_name,
        full_eval,
        validate_dataloader,
        testset_valid_temp,
        holdout_valid_temp,
        True,
        30,
    )
    del testset_valid_temp
    del holdout_valid_temp
    #

    val_metrics = trainer.train()
    torch.save(model, f"models/{exp_name}.pt")

    # calc successive_metrics
    # data_description_temp = {
    #     "users": "userid",
    #     "items": "itemid",
    #     "order": "timestamp",
    #     "n_users": 5400,
    #     "n_items": 3658,
    # }
    data_description_temp = {
        "users": "userid",
        "items": "itemid",
        "order": "timestamp",
        "n_users": 268531,
        "n_items": 128804,
    }

    test_sequences = data_to_sequences(
        testset.rename(columns=inverse_columns_mapping), data_description_temp
    )
    del testset

    all_metrics = {
        "leave_one_out": val_metrics,
    }
    if calc_successive:
        all_metrics["successive_metrics"] = calc_successive_metrics(
            model, test_sequences, data_description_temp, torch.device("cuda")
        )

    with open(Path("experiments") / (exp_name + ".json"), "+w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
