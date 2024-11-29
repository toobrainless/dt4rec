import numpy as np
import pandas as pd


def transform_indices(data, users, items):
    data_index = {}
    for entity, field in zip(["users", "items"], [users, items]):
        idx, idx_map = to_numeric_id(data, field)
        data_index[entity] = idx_map
        data.loc[:, field] = idx
    return data, data_index


def to_numeric_id(data, field):
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map


def topidx(a, topn):
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]


def topn_recommendations(scores, topn=10):
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations


def downvote_seen_items(scores, data, data_description):
    userid = data_description["users"]
    itemid = data_description["items"]
    # get indices of observed data
    user_idx = data[userid].values
    item_idx = data[itemid].values
    # downvote scores at the corresponding positions
    user_idx, _ = pd.factorize(user_idx, sort=True)
    seen_idx_flat = np.ravel_multi_index((user_idx, item_idx), scores.shape)
    np.put(scores, seen_idx_flat, -np.inf)


def calculate_topn_metrics(
    recommended_items, holdout_items, n_items, n_test_users, topn
):
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)

    # HR calculation
    hr = np.mean(hits_mask.any(axis=1))

    # MRR calculation
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1 / hit_rank) / n_test_users

    # NDCG calculation
    ndcg = np.sum(1 / np.log2(hit_rank + 1.0)) / n_test_users

    # COV calculation
    cov = np.unique(recommended_items[:, :topn]).size / n_items

    return {"hr": hr, "mrr": mrr, "ndcg": ndcg, "cov": cov}


def model_evaluate(
    recommended_items, holdout, holdout_description, topn_list=(10)
):
    n_items = holdout_description["n_items"]
    itemid = holdout_description["items"]
    holdout_items = holdout.sort_values(["user_idx"])[itemid].values
    n_test_users = recommended_items.shape[0]
    assert recommended_items.shape[0] == len(holdout_items)

    metrics = {}

    for topn in topn_list:
        new_metrics = {
            f"{key}@{topn}": value
            for key, value in calculate_topn_metrics(
                recommended_items, holdout_items, n_items, n_test_users, topn
            ).items()
        }
        metrics.update(new_metrics)

    return metrics
