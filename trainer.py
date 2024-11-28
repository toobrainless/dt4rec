import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as func
from tqdm import tqdm

from utils import (calc_leave_one_out_full, calc_leave_one_out_partial,
                   calc_metrics)

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
        full_eval,
        validate_dataloader=None,
        train_df=None,
        test_df=None,
        use_cuda=True,
        validation_num_batch=None,
    ):
        self.exp_name = exp_name
        (Path("models") / exp_name).mkdir(exist_ok=True)
        self.metrics = []
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = tconf.optimizer
        self.epochs = tconf.epochs
        self.lr_scheduler = tconf.lr_scheduler

        self.validate_dataloader = validate_dataloader
        self.train_df = train_df
        self.test_df = test_df
        self.fulll_eval = full_eval

        # take over whatever gpus are on the system
        self.device = "cpu"
        if use_cuda and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            # self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model = self.model.to("cuda")
        if validation_num_batch is None:
            self.validation_num_batch = len(validate_dataloader)
        else:
            self.validation_num_batch = validation_num_batch

    def _move_batch(self, batch):
        return [elem.to(self.device) for elem in batch]

    def _train_epoch(self, epoch):
        self.model.train()

        losses = []
        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc="_train_epoch",
        )

        for iter_, batch in pbar:
            if (iter_ + 1) % 1000 == 0:
                self._evalutation_epoch()
                print(self.metrics[-1])
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

        return np.mean(losses)

    def _evalutation_epoch(self):
        if self.fulll_eval:
            metrics = calc_leave_one_out_full(
                self.model, self.validate_dataloader, self.train_df, self.test_df
            )
        else:
            metrics = calc_leave_one_out_partial(
                self.model,
                self.validate_dataloader,
                self.validation_num_batch,
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
            torch.save(self.model, f"models/{self.exp_name}/epoch{epoch}.pickle")
            if self.validate_dataloader is not None:
                self._evalutation_epoch()
            self.metrics[-1]["loss"] = loss
            self.metrics[-1]["epoch_time"] = end - start
            print(self.metrics[-1])
        return self.metrics
