import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as func
from tqdm import tqdm

from utils import calc_leave_one_out, inf_loop

logger = logging.getLogger(__name__)


class TrainerConfig:
    """
    Config holder for trainer
    """

    epochs = 1
    lr_scheduler = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, **kwargs):
        """
        Arguments setter
        """
        for key, value in kwargs.items():
            setattr(self, key, value)


class Trainer:
    """
    Trainer for DT4Rec
    """

    grad_norm_clip = 1.0

    def __init__(
        self,
        model,
        train_dataloader,
        tconf,
        exp_name,
        checkpoints=False,
        validate_dataloader=None,
        train_df=None,
        test_df=None,
        use_cuda=True,
        len_epoch=None,
    ):
        self.exp_name = exp_name
        self.checkpoints = checkpoints
        if self.checkpoints:
            (Path("models") / exp_name).mkdir(exist_ok=True)
        self.metrics = []
        self.model = model
        self.train_dataloader = inf_loop(train_dataloader)
        self.optimizer = tconf.optimizer
        self.epochs = tconf.epochs
        self.lr_scheduler = tconf.lr_scheduler

        self.validate_dataloader = validate_dataloader
        self.train_df = train_df
        self.test_df = test_df
        self.len_epoch = len_epoch

        # take over whatever gpus are on the system
        self.device = "cpu"
        if use_cuda and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            # self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model = self.model.to("cuda")

    def _move_batch(self, batch):
        return [elem.to(self.device) for elem in batch]

    def _train_epoch(self, epoch):
        self.model.train()

        losses = []
        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=self.len_epoch,
            desc="_train_epoch",
        )

        for iter_, batch in pbar:
            # place data on the correct device
            states, actions, rtgs, timesteps, users = (
                batch["states"],
                batch["actions"],
                batch["rtgs"],
                batch["timesteps"],
                batch["users"],
            )
            states, actions, rtgs, timesteps, users = self._move_batch(
                [states, actions, rtgs, timesteps, users]
            )
            targets = actions

            # forward the model
            logits = self.model(states, actions, rtgs, timesteps, users)

            loss = func.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            ).mean()
            losses.append(loss.item())

            # backprop and update the parametersx
            self.model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if iter_ >= self.len_epoch:
                break

        return np.mean(losses)

    def _evalutation_epoch(self):
        metrics = calc_leave_one_out(
            self.model,
            self.validate_dataloader,
            self.train_df,
            self.test_df,
        )

        self.metrics.append(metrics)

    def train(self):
        """
        Run training loop
        """
        for epoch in range(self.epochs):
            start = time.time()
            loss = self._train_epoch(epoch)
            end = time.time()
            if self.checkpoints:
                torch.save(self.model, f"models/{self.exp_name}/epoch{epoch}.pt")
            if self.validate_dataloader is not None:
                self._evalutation_epoch()
            self.metrics[-1]["loss"] = loss
            self.metrics[-1]["epoch_time"] = end - start
            print(self.metrics[-1])
        return self.metrics
