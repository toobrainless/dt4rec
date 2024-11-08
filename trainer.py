import logging

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as func
from tqdm import tqdm

from utils import calc_metrics

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
        validate_dataloader=None,
        train_df=None,
        test_df=None,
        use_cuda=True,
    ):
        self.metrics = []
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = tconf.optimizer
        self.epochs = tconf.epochs
        self.lr_scheduler = tconf.lr_scheduler

        self.validate_dataloader = validate_dataloader
        self.train_df = train_df
        self.test_df = test_df

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
            total=len(self.train_dataloader),
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

    def _evaluation_epoch(self):
        self.model.eval()
        item_num = self.model.config.vocab_size - 1
        user_num = self.model.user_num
        logits = np.zeros((user_num, item_num))

        for idx, batch in tqdm(
            enumerate(self.validate_dataloader), total=len(self.validate_dataloader)
        ):
            batch = {key: value.to("cuda") for key, value in batch.items()}
            output = self.model(**batch)[:, -1, :-1].detach().cpu().numpy()
            batch_size = output.shape[0]
            logits[idx * batch_size : (idx + 1) * batch_size] = output

        metrics = calc_metrics(logits, self.train_df, self.test_df)
        print(metrics)
        self.metrics.append(metrics)

    def train(self):
        """
        Run training loop
        """
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            if self.validate_dataloader is not None:
                self._evaluation_epoch()
        return self.metrics
