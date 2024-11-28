import click
import numpy as np
import torch
from train import get_ds
from utils import calc_leave_one_out


torch.manual_seed(41)
np.random.seed(41)


@click.command()
@click.argument("model_name")
@click.argument("ds_name")
@click.option("--validate_batch_size", "-vbs", default=128)
def main(model_name, ds_name, validate_batch_size):
    model = torch.load(f"models/{model_name}.pt")
    trajectory_len = model.block_size // 3
    (
        train,
        holdout,
        validate_dataloader,
        _,
        _,
        _,
    ) = get_ds(ds_name, trajectory_len, validate_batch_size, None, False)

    metrics = calc_leave_one_out(model, validate_dataloader, train, holdout)
    print(metrics)


if __name__ == "__main__":
    main()
