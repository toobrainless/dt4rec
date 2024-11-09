import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gpt1 import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from utils import (
    LeaveOneOutDataset,
    WarmUpScheduler,
    calc_successive_metrics,
    data_to_sequences,
    get_dataloader,
)


def main():
    # some params
    argv = sys.argv
    for arg in argv[2:]:
        assert arg in ["True", "False"]
    exp_name = argv[1]
    use_svd_embs = argv[2] == "True"
    learn_svd_embs = argv[3] == "True"

    trajectory_len = 30

    # get and fix Danil get_dataset
    columns_mapping = {
        "userid": "user_idx",
        "itemid": "item_idx",
        "rating": "relevance",
        "timestamp": "timestamp",
    }
    inverse_columns_mapping = {value: key for key, value in columns_mapping.items()}

    def read_and_rename(path):
        return pd.read_csv(path).rename(columns=columns_mapping)

    training_temp = read_and_rename("data_split/training_temp.csv")
    testset_valid_temp = read_and_rename("data_split/testset_valid_temp.csv")
    testset = read_and_rename("data_split/testset.csv")
    holdout_valid_temp = read_and_rename("data_split/holdout_valid_temp.csv")
    #

    item_num = testset_valid_temp["item_idx"].max() + 1
    user_num = testset_valid_temp["user_idx"].max() + 1

    # create validate_datalaoder
    last_df = (
        testset_valid_temp.sort_values(["user_idx", "timestamp"])
        .groupby("user_idx")
        .tail(trajectory_len - 1)
    )
    validate_dataset = LeaveOneOutDataset(last_df, user_num, item_num)
    batch_size = 128
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=batch_size,
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
    if use_svd_embs:
        item_embs = np.load("/home/hdilab/amgimranov/my_dt4rec/item_embs_ilya.npy")
        model.state_repr.item_embeddings.weight.data = torch.from_numpy(item_embs)
        if not learn_svd_embs:
            model.state_repr.item_embeddings.weight.requires_grad = False
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
        validate_dataloader,
        testset_valid_temp,
        holdout_valid_temp,
        True,
    )
    #

    val_metrics = trainer.train()

    # calc successive_metrics
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
    successive_metrics = calc_successive_metrics(
        model, test_sequences, data_description_temp
    )

    all_metrics = {
        "leave_one_out": val_metrics,
        "successive_metrics": successive_metrics,
    }

    with open(Path("experiments") / (exp_name + ".json"), "+w") as f:
        json.dump(all_metrics, f, indent=2)
    torch.save(model, f"models/{exp_name}.pt")


if __name__ == "__main__":
    main()
