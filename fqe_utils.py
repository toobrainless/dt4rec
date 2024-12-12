"""
This module provides a set of functions and utilities for computing FQE (Finite-Horizon Q Evaluation).
It includes tools for data preparation, state embedding calculation, and scoring.
"""

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = torch.device("cuda")


def batch_to_device(batch):
    """
    Move the batch of tensors to the specified device (e.g., GPU).
    """
    new_batch = {key: value.to(DEVICE) for key, value in batch.items()}
    return new_batch


def make_rsa(item_seq, memory_size, item_num, inference=False):
    """
    Create a dictionary of Reinforcement Learning (RL)-specific tensors (states, actions, rewards, etc.)
    based on the input sequence. Supports both inference and training modes.
    """
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


class SeqsDataset(Dataset):

    def __init__(self, seqs, item_num):
        """
        Initialize the dataset with a list of sequences and the total number of items.
        """
        self.seqs = seqs
        self.item_num = item_num

    def __getitem__(self, idx):
        """
        Retrieve a single sequence and convert it into RL-specific tensors (e.g., states, actions, rewards).
        """
        seq = self.seqs[idx]
        if len(seq) == 0:
            seq = torch.tensor([self.item_num, self.item_num, self.item_num])
        return make_rsa(self.seqs[idx], 3, self.item_num, True)

    def __len__(self):
        """
        Return the number of sequences in the dataset.
        """
        return len(self.seqs)


def seq_to_states(model, seqs):
    """
    Convert a list of sequences into a batch of state embeddings using the model's state representation function.
    """
    item_num = model.config.vocab_size
    seqs_dataset = SeqsDataset(seqs, item_num)
    seqs_dataloader = DataLoader(seqs_dataset, batch_size=128, num_workers=4)

    outputs = []
    # for batch in tqdm(seqs_dataloader, total=len(seqs_dataloader)):
    for batch in seqs_dataloader:
        batch = batch_to_device(batch)
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


def score_with_state(model, seq):
    """
    Compute the state embedding for a single input sequence and return it.
    """
    model.eval()
    item_num = model.config.vocab_size
    seq_len = 100
    seq = torch.LongTensor([1, 2, 3, 4])
    seq = F.pad(seq, (seq_len - 1 - len(seq), 0), value=item_num)
    rsa = {key: value[None, ...] for key, value in make_rsa(seq, 3, item_num).items()}
    state = model.calc_hidden_state(**rsa).detach()[:, -1].cpu()

    return state


def score_batch(model, seqs):
    """
    Compute model scores (logits) for a batch of sequences, using a DataLoader for efficiency.
    """
    model.eval()
    item_num = model.config.vocab_size
    seqs_dataset = SeqsDataset(seqs, item_num)
    seqs_dataloader = DataLoader(seqs_dataset, batch_size=128, num_workers=4)

    outputs = []
    # for batch in tqdm(seqs_dataloader, total=len(seqs_dataloader)):
    for batch in seqs_dataloader:
        batch = batch_to_device(batch)
        outputs.append(model(**batch).detach()[:, -1])

    return torch.concat(outputs, dim=0)


def my_collate(batch):
    """
    Custom collate function for batching list of dicts into a single batch.
    """
    output = {key: [] for key in batch[0].keys()}
    for elem in batch:
        for key, value in elem.items():
            output[key].append(value)
    for key, value in output.items():
        output[key] = torch.stack(value)
    return output


def score_batch2(model, seqs):
    """
    Compute model scores (logits) for a batch of sequences using a manual batching approach.
    """
    model.eval()
    item_num = model.config.vocab_size
    batch = my_collate([make_rsa(seq, 3, item_num, True) for seq in seqs])
    batch = batch_to_device(batch)
    with torch.no_grad():
        logits = model(**batch).detach()[:, -1]
    return logits


def state_batch2(model, seqs):
    """
    Compute state embeddings for a batch of sequences using a manual batching approach.
    """
    item_num = model.config.vocab_size
    batch = my_collate([make_rsa(seq, 3, item_num, True) for seq in seqs])
    batch = batch_to_device(batch)

    with torch.no_grad():
        hidden_state = model.calc_hidden_state(**batch).detach()[:, -1].cpu()
    return hidden_state


def state_batch(model, seqs, use_tqdm=False):
    """
    Compute state embeddings for a batch of sequences, optionally using a progress bar (via tqdm).
    """
    item_num = model.config.vocab_size
    seqs_dataset = SeqsDataset(seqs, item_num)
    seqs_dataloader = DataLoader(seqs_dataset, batch_size=128, num_workers=4)

    outputs = []
    iterator = (
        tqdm(seqs_dataloader, total=len(seqs_dataloader))
        if use_tqdm
        else seqs_dataloader
    )
    with torch.no_grad():
        for batch in iterator:
            batch = batch_to_device(batch)

            hidden_state = model.calc_hidden_state(**batch).detach()[:, -1].cpu()
            outputs.append(hidden_state)
            del batch, hidden_state

    return torch.concat(outputs, dim=0)
