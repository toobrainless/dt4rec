import os
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from absl import app, flags
from data import data_to_sequences, data_to_sequences_rating, get_dataset
from eval_utils import get_test_scores, model_evaluate, sasrec_model_scoring
from ml_collections import config_flags
from rl_ope.utils import (extract_states_actions, extract_states_actions_val,
                          prepare_svd)
from tqdm import tqdm
from train import build_sasrec_model, prepare_sasrec_model

from utils import downvote_seen_items, topn_recommendations

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"


def recommend_sequential(
    model,
    target_seq: Union[list, np.ndarray],
    seen_seq: Union[list, np.ndarray],
    topn: int,
    *,
    user: Optional[int] = None,
    item_embs,
):
    """Given an item sequence and a sequence of next target items,
    predict top-n candidates for each next step in the target sequence.
    """
    model.eval()
    predictions = predict_sequential(
        model, target_seq[:-1], seen_seq, user=user, item_embs=item_embs
    )
    predictions[:, seen_seq] = -np.inf
    for k in range(1, predictions.shape[0]):
        predictions[k, target_seq[:k]] = -np.inf
    predicted_items = np.apply_along_axis(topidx, 1, predictions, topn)
    return predicted_items


def predict_sequential(
    model, target_seq, seen_seq, user, item_embs
):  # example for SASRec

    maxlen = 100  # тут длина контекста сасрека

    n_seen = len(seen_seq)
    n_targets = len(target_seq)
    seq = np.concatenate([seen_seq, target_seq])

    with torch.no_grad():
        pad_seq = torch.as_tensor(
            np.pad(
                seq,
                (max(0, maxlen - n_seen), 0),
                mode="constant",
                constant_values=model.pad_token,
            ),
            dtype=torch.int64,
            device=device,
        )
        log_seqs = torch.as_strided(
            pad_seq[-n_targets - maxlen :], (n_targets + 1, maxlen), (1, 1)
        )
        log_feats = model.log2feats(log_seqs, item_embs)[:, -1, :]

        if item_embs is None:
            item_embs = model.item_emb.weight

        logits = item_embs.matmul(log_feats.unsqueeze(-1)).squeeze(-1)

    return logits.detach().cpu().numpy()


def topidx(arr, topn):
    parted = np.argpartition(arr, -topn)[-topn:]
    return parted[np.argsort(-arr[parted])]


def main():
    n_neg_samples = -1
    data_path = "/home/hdilab01/IlyaRecSys/RECEsvd/mv1m/ml-1m.zip"

    # training_temp, data_description_temp, testset_valid_temp, _, holdout_valid_temp, _ = get_dataset(local_file=data_path,
    #                                                                                      splitting='temporal_full',
    #                                                                                      q=0.8)

    # training_full, data_description_full, testset_valid_full, _, holdout_valid_full, _ = get_dataset(local_file=data_path,
    #                                                                                      splitting='full',
    #                                                                                      q=0.8)

    # test_sequences = data_to_sequences_rating(testset_valid_full, data_description_full, n_neg_samples)

    (
        training_temp,
        data_description_temp,
        testset_valid_temp,
        testset,
        holdout_valid_temp,
        _,
    ) = get_dataset(local_file=data_path, splitting="temporal_full", q=0.8)
    test_sequences = data_to_sequences(testset, data_description_temp)

    s = training_temp["rating"]
    training_temp["rating"] = s.where(s >= 3, 0).mask(s >= 3, 1)

    model_D_conf = {
        "manual_seed": 123,
        "sampler_seed": 123,
        "use_svd_emb": False,
        "num_epochs": 100,
        "maxlen": 100,
        "hidden_units": 64,
        "dropout_rate": 0.3,
        "num_blocks": 2,
        "num_heads": 1,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "fwd_type": "ce",
        "l2_emb": 0,
        "patience": 10,
        "skip_epochs": 1,
        "n_neg_samples": 0,
        "sampling": "no_sampling",
    }

    model_e0_conf = {
        "manual_seed": 123,
        "sampler_seed": 123,
        "use_svd_emb": False,
        "num_epochs": 100,
        "maxlen": 100,
        "hidden_units": 64,
        "dropout_rate": 0.3,
        "num_blocks": 2,
        "num_heads": 1,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "fwd_type": "ce",
        "l2_emb": 0,
        "patience": 10,
        "skip_epochs": 1,
        "n_neg_samples": 0,
        "sampling": "no_sampling",
    }

    model_e1_conf = {
        "manual_seed": 123,
        "sampler_seed": 123,
        "use_svd_emb": False,
        "num_epochs": 10,
        "maxlen": 100,
        "hidden_units": 64,
        "dropout_rate": 0.3,
        "num_blocks": 2,
        "num_heads": 1,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "fwd_type": "ce",
        "l2_emb": 0,
        "patience": 10,
        "skip_epochs": 1,
        "n_neg_samples": 0,
        "sampling": "no_sampling",
    }

    model_e2_conf = {
        "manual_seed": 123,
        "sampler_seed": 123,
        "use_svd_emb": False,
        "num_epochs": 20,
        "maxlen": 100,
        "hidden_units": 32,
        "dropout_rate": 0.9,
        "num_blocks": 2,
        "num_heads": 1,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "fwd_type": "ce",
        "l2_emb": 0,
        "patience": 10,
        "skip_epochs": 1,
        "n_neg_samples": 0,
        "sampling": "no_sampling",
    }

    model_D_sasrec, _, _, _ = prepare_sasrec_model(
        model_D_conf, training_temp, data_description_temp, device
    )
    model_D_sasrec.load_state_dict(
        torch.load(
            "./saved_models/model_D_sasrec.pt", map_location=torch.device(device)
        )
    )

    model_e0, _, _, _ = prepare_sasrec_model(
        model_e0_conf, training_temp, data_description_temp, device
    )
    model_e0.load_state_dict(
        torch.load("./saved_models/model_e0.pt", map_location=torch.device(device))
    )

    model_e1, _, _, _ = prepare_sasrec_model(
        model_e1_conf, training_temp, data_description_temp, device
    )
    model_e1.load_state_dict(
        torch.load("./saved_models/model_e1.pt", map_location=torch.device(device))
    )

    model_e2, _, _, _ = prepare_sasrec_model(
        model_e2_conf, training_temp, data_description_temp, device
    )
    model_e2.load_state_dict(
        torch.load("./saved_models/model_e2.pt", map_location=torch.device(device))
    )

    item_embs = prepare_svd(training_temp, data_description_temp, 64, device)

    models = {
        "SASRec 1": (model_D_sasrec, None),
        "SASRec 2": (model_e0, None),
        "SASRec 3": (model_e1, None),
        "SASRec 4": (model_e2, None),
        "SasRec 5": (model_D_sasrec, item_embs),
    }

    topn = 10
    results_list = []

    for model_name, (model, item_embs_var) in models.items():
        val_scores = sasrec_model_scoring(
            model, testset_valid_temp, data_description_temp, item_embs_var, device
        )
        downvote_seen_items(val_scores, testset_valid_temp, data_description_temp)
        val_recs = topn_recommendations(val_scores, topn=topn)
        val_metrics = model_evaluate(
            val_recs, holdout_valid_temp, data_description_temp
        )
        ndcg = val_metrics[f"ndcg@{topn}"]
        hr = val_metrics[f"hr@{topn}"]
        mrr = val_metrics[f"mrr@{topn}"]
        cov = val_metrics[f"cov@{topn}"]

        results_list.append(
            pd.DataFrame(
                data={"score": [hr, mrr, ndcg, cov]},
                index=[f"{metric}@{topn}" for metric in ["HR", "MRR", "NDCG", "COV"]],
                # columns=[model_name]  # Label results by model name
            )
        )
        # val_metrics = {"ndcg@10": val_metrics["ndcg@10"],
        #                "hr@10": val_metrics["hr@10"],
        #                "mrr@10": val_metrics["mrr@10"],
        #                "cov@10": val_metrics["cov@10"]}

    final_results = pd.concat(results_list, axis=1)
    final_results.columns = models.keys()
    print(final_results)

    results_list = []

    for model_name, (model, item_embs_var) in models.items():
        cum_hits = 0
        cum_reciprocal_ranks = 0.0
        cum_discounts = 0.0
        unique_recommendations = set()
        total_count = 0
        cov = []

        # Loop over each user and test sequence
        for user, test_seq in tqdm(test_sequences.items(), total=len(test_sequences)):
            seen_seq = test_seq[:1]
            test_seq = test_seq[1:]
            num_predictions = len(test_seq)

            if not num_predictions:  # if no test items left - skip user
                continue

            # Get predicted items
            predicted_items = recommend_sequential(
                model, test_seq, seen_seq, topn, user=user, item_embs=item_embs_var
            )

            # compute hit steps and indices
            # print(predicted_items == np.atleast_2d(test_seq).T)
            hit_steps, hit_index = np.nonzero(
                predicted_items == np.atleast_2d(test_seq).T
            )

            cov.append(
                len(np.unique(predicted_items.ravel()))
                / data_description_temp["n_items"]
            )

            num_hits = hit_index.size
            if num_hits:
                cum_hits += num_hits
                cum_reciprocal_ranks += np.sum(1.0 / (hit_index + 1))
                cum_discounts += np.sum(1.0 / np.log2(hit_index + 2))

            total_count += num_predictions

        # evaluation metrics for the current model
        hr = cum_hits / total_count
        mrr = cum_reciprocal_ranks / total_count
        dcg = cum_discounts / total_count
        cov = np.mean(cov)

        results_list.append(
            pd.DataFrame(
                data={"score": [hr, mrr, dcg, cov]},
                index=[f"{metric}@{topn}" for metric in ["HR", "MRR", "NDCG", "COV"]],
                # columns=[model_name]  # Label results by model name
            )
        )

    final_results = pd.concat(results_list, axis=1)
    final_results.columns = models.keys()
    print(final_results)


if __name__ == "__main__":
    # python run_fqe.py --config=config.py:SASRec --config.config_e.chkpt_path=./saved_models/model_e0.pt --config.values_path=./saved_values/values_0.npy
    # main()
    main()
