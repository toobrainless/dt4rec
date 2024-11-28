import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gpt1 import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from utils import (
    LeaveOneOutDataset,
    SeqsDataset,
    calc_successive_metrics,
    data_to_sequences,
    get_all_seqs,
    set_seed,
)

set_seed(41)

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


def get_ds(
    ds_name, trajectory_len, validate_batch_size, train_batch_size, return_train
):
    assert ds_name in ["movielens", "zvuk_danil", "zvuk_my_split"]
    data_folder = Path(f"data_split/{ds_name}")

    if ds_name == "movielens":
        train = read_and_rename(data_folder / "testset_valid_temp.csv", use_csv=True)
        test = read_and_rename(data_folder / "testset.csv", use_csv=True)
        holdout = read_and_rename(data_folder / "holdout_valid_temp.csv", use_csv=True)
    if ds_name == "zvuk_danil":
        train = read_and_rename("data_split/zvuk_danil/training_temp.parquet")
        test = read_and_rename("data_split/zvuk_danil/testset.parquet")
        holdout = read_and_rename("data_split/zvuk_danil/holdout_valid_temp.parquet")

    item_num = train.item_idx.max() + 1
    user_num = train.user_idx.max() + 1

    # create validate_datalaoder
    validate_dataset = LeaveOneOutDataset(train, holdout, trajectory_len)
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=validate_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    # create train_dataloader
    if return_train:
        if (data_folder / "all_seqs.pt").exists():
            print("all_seqs exists...")
            all_seqs = torch.load(data_folder / "all_seqs.pt")
        else:
            print("getting all_seqs...")
            all_seqs = get_all_seqs(
                train, trajectory_len, train.item_idx.max() + 1, None
            )
            torch.save(all_seqs, data_folder / "all_seqs.pt")

        train_dataloader = DataLoader(
            SeqsDataset(all_seqs, memory_size=3, item_num=item_num),
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
    else:
        train_dataloader = None

    return (
        train,
        holdout,
        validate_dataloader,
        train_dataloader,
        item_num,
        user_num,
    )


@click.command()
@click.argument("exp_name")
@click.argument(
    "ds_name",
)
@click.option("--train_batch_size", "-tbs", default=128)
@click.option("--validate_batch_size", "-vbs", default=128)
@click.option("--use_svd", default=False)
@click.option("--learn_svd", default=False)
@click.option("--trajectory_len", "-tl", default=100)
@click.option("--calc_successive", "-cs", default=False)
@click.option("--len_epoch", "-le", default=1000)
@click.option("--num_epoch", "-ne", default=10)
@click.option("--checkpoints", "-c", is_flag=True)
def main(
    exp_name,
    ds_name,
    train_batch_size,
    validate_batch_size,
    use_svd,
    learn_svd,
    trajectory_len,
    calc_successive,
    len_epoch,
    num_epoch,
    checkpoints,
):
    (
        train,
        holdout,
        validate_dataloader,
        train_dataloader,
        item_num,
        user_num,
    ) = get_ds(ds_name, trajectory_len, validate_batch_size, train_batch_size, True)

    print(f"{len_epoch / len(train_dataloader)=}")

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
    print(f"param num: {total_params}")

    if use_svd:
        item_embs = np.load("/home/hdilab/amgimranov/dt4rec/item_embs_ilya.npy")
        model.state_repr.item_embeddings.weight.data = torch.from_numpy(item_embs)
        model.state_repr.item_embeddings.weight.requires_grad = learn_svd
    #

    # create trainer
    tconf = TrainerConfig(epochs=num_epoch)

    optimizer = torch.optim.AdamW(
        model.configure_optimizers(),
        lr=3e-4,
        betas=(0.9, 0.95),
    )
    # lr_scheduler = WarmUpScheduler(optimizer, dim_embed=768, warmup_steps=4000)
    lr_scheduler = None

    tconf.update(optimizer=optimizer, lr_scheduler=lr_scheduler)
    trainer = Trainer(
        model,
        train_dataloader,
        tconf,
        exp_name,
        checkpoints,
        validate_dataloader,
        train,
        holdout,
        True,
        len_epoch,
    )
    del train
    del holdout
    #

    val_metrics = trainer.train()
    torch.save(model, f"models/{exp_name}.pt")

    all_metrics = {
        "leave_one_out": val_metrics,
    }
    if calc_successive:
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
        all_metrics["successive_metrics"] = calc_successive_metrics(
            model, test_sequences, data_description_temp, torch.device("cuda")
        )

    with open(Path("experiments") / (exp_name + ".json"), "+w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
