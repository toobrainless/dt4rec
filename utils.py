import bisect
import random
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from rs_datasets import MovieLens
from sklearn.preprocessing import LabelEncoder
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from metrics import Evaluator


def set_seed(seed):
    """
    Set random seed in all dependicies
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WarmUpScheduler(_LRScheduler):
    """
    Implementation of WarmUp
    """

    def __init__(
        self,
        optimizer: Optimizer,
        dim_embed: int,
        warmup_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups


def calc_lr(step, dim_embed, warmup_steps):
    """
    Learning rate calculation
    """
    return dim_embed ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def time_split(
    df: pd.DataFrame,
    timestamp_col,
    train_size,
    drop_cold_items,
    drop_cold_users,
    item_col,
    user_col,
):
    df = df.sort_values(timestamp_col)
    train_len = int(len(df) * train_size)
    train = df.iloc[:train_len]
    test = df.iloc[train_len:]

    if drop_cold_items:
        test = test[test[item_col].isin(train[item_col])]

    if drop_cold_users:
        test = test[test[user_col].isin(train[user_col])]

    return train, test


def split_last_n(_df, user_col, item_col, n=1, drop_cold=True):
    df = _df.copy()
    df = df.sort_values([user_col, "timestamp"])
    df["row_num"] = df.groupby(user_col).cumcount() + 1
    df["count"] = df.groupby(user_col)[user_col].transform(len)
    df["is_test"] = df["row_num"] > (df["count"] - float(n))
    df = df.drop(columns=["row_num", "count"])
    train = df[~df.is_test].drop(columns=["is_test"])
    test = df[df.is_test].drop(columns=["is_test"])
    if drop_cold:
        test = test[test[item_col].isin(train[item_col])]
        test = test[test[user_col].isin(train[user_col])]
        train = train[train[user_col].isin(test[user_col])]

    return train.reset_index(drop=True), test.reset_index(drop=True)


class MyIndexer:
    def __init__(self, user_col, item_col):
        self.user_col = user_col
        self.item_col = item_col
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    def fit(self, X):
        self.user_encoder.fit(X[self.user_col])
        self.item_encoder.fit(X[self.item_col])

        return self

    def transform(self, X):
        X[self.user_col] = self.user_encoder.transform(X[self.user_col])
        X[self.item_col] = self.item_encoder.transform(X[self.item_col])

        return X

    def fit_transform(self, X):
        old_len_items = len(set(X[self.item_col]))
        old_len_users = len(set(X[self.user_col]))
        ans = self.fit(X).transform(X)
        assert (
            old_len_items
            == len(set(ans[self.item_col]))
            == (ans[self.item_col].max() - ans[self.item_col].min() + 1)
        )
        assert (
            old_len_users
            == len(set(ans[self.user_col]))
            == (ans[self.user_col].max() - ans[self.user_col].min() + 1)
        )
        return ans


# dataset stuff


class SeqsDataset(Dataset):
    def __init__(self, seqs, memory_size, item_num):
        self.memory_size = memory_size
        self.seqs = seqs
        self.item_num = item_num

    def __getitem__(self, idx):
        return make_rsa(
            self.seqs[idx], memory_size=self.memory_size, item_num=self.item_num
        )

    def __len__(self):
        return len(self.seqs)


def make_rsa(item_seq, memory_size, item_num, inference=False):
    if inference:
        return {
            "rtgs": torch.arange(len(item_seq) + 1, 0, -1)[..., None],
            "states": F.pad(item_seq, (memory_size, 0), value=item_num).unfold(0, 3, 1),
            "actions": item_seq[..., None],
            "timesteps": torch.tensor([[0]]),
            "users": torch.tensor([0]),
        }
    return {
        "rtgs": torch.arange(len(item_seq), 0, -1)[..., None],
        "states": F.pad(item_seq, (memory_size, 0), value=item_num).unfold(0, 3, 1)[
            :-1
        ],
        "actions": item_seq[..., None],
        "timesteps": torch.tensor([[0]]),
        "users": torch.tensor([0]),
    }


def get_seqs(group: pd.DataFrame, seq_len, pad_value):
    items = torch.from_numpy(group["item_idx"].to_numpy())
    items = F.pad(items, (seq_len, 0), value=pad_value)
    return items.unfold(0, seq_len, 1)


def get_all_seqs(df, seq_len, pad_value, user_num):
    all_seqs = []

    for user in trange(user_num):
        user_df = df[df.user_idx == user]
        user_seqs = get_seqs(user_df, seq_len, pad_value)
        all_seqs.append(user_seqs)

    all_seqs = torch.concat(all_seqs, dim=0)

    return all_seqs


def get_dataloader(df, memory_size, seq_len, pad_value, user_num, item_num):
    df = df.sort_values(["user_idx", "timestamp"])
    df.groupby(by=["user_idx"])

    all_seqs = get_all_seqs(df, seq_len, pad_value, user_num)

    dataloader = DataLoader(
        SeqsDataset(all_seqs, memory_size=memory_size, item_num=item_num),
        128,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    return dataloader


class LeaveOneOutDataset:
    def __init__(self, last_df, user_num, item_num):
        self.last_df = last_df
        self.user_num = user_num
        self.item_num = item_num

    def __getitem__(self, user):
        items = torch.from_numpy(
            self.last_df[self.last_df.user_idx == user].item_idx.to_numpy()
        )
        items = F.pad(items, (29 - len(items), 0), value=self.item_num)
        rsa = make_rsa(items, 3, True)
        rsa["rtgs"][0, -1] = 10

        return rsa

    def __len__(self):
        return self.user_num


def calc_metrics(logits, train, test):
    evaluator = Evaluator(top_k=[10])
    scores_downvoted = evaluator.downvote_seen_items(
        logits,
        train.rename(
            columns={"user_idx": "userid", "item_idx": "itemid", "relevance": "rating"}
        ),
    )
    recs = evaluator.topk_recommendations(scores_downvoted)
    metrics = evaluator.compute_metrics(
        test.rename(
            columns={"user_idx": "userid", "item_idx": "itemid", "relevance": "rating"}
        ),
        recs,
        train.rename(
            columns={"user_idx": "userid", "item_idx": "itemid", "relevance": "rating"}
        ),
    )

    return metrics


def get_dataset(drop_bad_ratings=False):
    df = MovieLens("1m").ratings.rename(
        columns={
            "user_id": "user_idx",
            "item_id": "item_idx",
            "rating": "relevance",
            "timestamp": "timestamp",
        }
    )
    if drop_bad_ratings:
        df = df[df.relevance >= 3]  # ??? в гет датасет Данила так не делается

    train, test = split_last_n(df, "user_idx", "item_idx")

    indexer = MyIndexer(user_col="user_idx", item_col="item_idx")
    train = indexer.fit_transform(train).reset_index(drop=True)
    test = indexer.transform(test).reset_index(drop=True)

    item_num = train["item_idx"].max() + 1
    user_num = train["user_idx"].max() + 1

    last_df = train.sort_values(["user_idx", "timestamp"]).groupby("user_idx").tail(29)
    validate_dataset = LeaveOneOutDataset(last_df, user_num, item_num)
    batch_size = 128
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train, test, validate_dataloader, item_num, user_num


def calc_diff(train, test):
    train_users = set(train.user_idx)
    test_users = set(test.user_idx)

    train_diff_test = train_users.difference(test_users)
    test_diff_train = test_users.difference(train_users)

    return train_diff_test, test_diff_train


def fill_bad_users(train, test):
    train_diff_test, test_diff_train = calc_diff(train, test)

    for bad_user in train_diff_test:
        test.loc[len(test)] = [bad_user, 1113, 5, 998315055]

    for bad_user in test_diff_train:
        train.loc[len(train)] = [bad_user, 1008, 5, 956715569]


class SeqsDataset2(Dataset):
    def __init__(self, seqs, item_num):
        self.seqs = seqs
        self.item_num = item_num

    def __getitem__(self, idx):
        return make_rsa(self.seqs[idx], 3, self.item_num, True)

    def __len__(self):
        return len(self.seqs)


def seq_to_logits(model, seqs):
    item_num = model.config.vocab_size
    seqs_dataset = SeqsDataset2(seqs, item_num)
    seqs_dataloader = DataLoader(seqs_dataset, batch_size=128, num_workers=4)

    outputs = []
    for batch in tqdm(seqs_dataloader, total=len(seqs_dataloader)):
        outputs.append(model(**batch).detach()[:, -1])

    return torch.concat(outputs, dim=0)


def seq_to_states(model, seqs):
    item_num = model.config.vocab_size
    seqs_dataset = SeqsDataset2(seqs, item_num)
    seqs_dataloader = DataLoader(seqs_dataset, batch_size=128, num_workers=4)

    outputs = []
    for batch in tqdm(seqs_dataloader, total=len(seqs_dataloader)):
        trajectory_len = batch["states"].shape[1]
        state_embeddings = model.state_repr(
            batch["users"].repeat((1, trajectory_len)).reshape(-1, 1),
            batch["states"].reshape(-1, 3),
        )

        state_embeddings = state_embeddings.reshape(
            batch["states"].shape[0], batch["states"].shape[1], model.config.n_embd
        )
        outputs.append(state_embeddings[:, -1])

    return torch.concat(outputs, dim=0)


def data_to_sequences(data, data_description):
    userid = data_description["users"]
    itemid = data_description["items"]

    sequences = (
        data.sort_values([userid, data_description["order"]])
        .groupby(userid, sort=False)[itemid]
        .apply(list)
    )
    return sequences


def split_and_pad_tensor(tensor, pad_token, chunk_size=30):
    n = tensor.size(0)
    num_chunks = (n + chunk_size - 1) // chunk_size

    padded_size = num_chunks * chunk_size

    padded_tensor = torch.full((padded_size,), fill_value=pad_token)
    padded_tensor[-n:] = tensor

    chunks = padded_tensor.view(num_chunks, chunk_size)

    return chunks


def calc_successive_metrics(model, test_sequences, data_description_temp):
    pad_token = model.state_repr.item_embeddings.padding_idx
    item_num = pad_token
    seqs = torch.concat(
        [
            split_and_pad_tensor(torch.tensor(x), pad_token)
            for x in test_sequences.values
        ]
    )

    dataset = SeqsDataset(seqs, 3, item_num)
    dataloader = DataLoader(
        list(zip(dataset, seqs)),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    cum_hits = 0
    cum_reciprocal_ranks = 0.0
    cum_discounts = 0.0
    unique_recommendations = set()
    total_count = 0
    cov = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        labels = batch[1].numpy()
        batch = {key: value.to("cuda") for key, value in batch[0].items()}
        model.train()
        batch_logits = model(**batch).detach().cpu().numpy()

        for logits_idx, logits in enumerate(batch_logits):
            pad_idx = len(np.where(labels[logits_idx] == pad_token)[0])
            for recs_idx, recs in enumerate(logits):
                recs[labels[logits_idx, pad_idx:recs_idx]] = -torch.inf
        predicted_items = batch_logits.argsort(axis=-1)[:, :, -10:]
        # labels [batch_size x 30]
        # predicted_items [batch_size x 30 x 10]
        _, _, hit_index = np.where(predicted_items == labels[..., None])
        cov.append(
            len(np.unique(predicted_items.ravel())) / data_description_temp["n_items"]
        )
        hit_index = 9 - hit_index

        unique_recommendations |= set(np.unique(predicted_items).tolist())
        num_hits = hit_index.size
        if num_hits:
            cum_hits += num_hits
            cum_reciprocal_ranks += np.sum(1.0 / (hit_index + 1))
            cum_discounts += np.sum(1.0 / np.log2(hit_index + 2))

    total_count = sum(map(len, test_sequences.values))
    hr = cum_hits / total_count
    mrr = cum_reciprocal_ranks / total_count
    ndcg = cum_discounts / total_count
    danil_cov = np.mean(cov)
    cov = len(unique_recommendations) / data_description_temp["n_items"]

    return {"hr": hr, "mrr": mrr, "ndcg": ndcg, "cov": cov, "danil_cov": danil_cov}
