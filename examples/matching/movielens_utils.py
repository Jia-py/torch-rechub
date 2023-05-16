"""
    util function for movielens data.
"""

import collections
import numpy as np
import pandas as pd
from torch_rechub.utils.match import Annoy
from torch_rechub.basic.metric import topk_metrics
from collections import Counter
import tqdm


def match_evaluation(user_embedding, item_embedding, test_user, all_item, user_col='user_id', item_col='video_id',
                     raw_id_maps="/root/code/torch-rechub/examples/ranking/data/ali-ccp/saved/raw_id_maps.npy", topk=10):
    print("evaluate embedding matching on test data")
    annoy = Annoy(n_trees=10)
    annoy.fit(item_embedding)

    #for each user of test dataset, get ann search topk result
    print("matching for topk")
    user_map, item_map = np.load(raw_id_maps, allow_pickle=True)
    match_res = collections.defaultdict(dict)  # user id -> predicted item ids
    for user_id, user_emb in zip(test_user[user_col], user_embedding):
        if len(user_emb.shape)==2:
            #多兴趣召回
            items_idx = []
            items_scores = []
            for i in range(user_emb.shape[0]):
                temp_items_idx, temp_items_scores = annoy.query(v=user_emb[i], n=topk)  # the index of topk match items
                items_idx += temp_items_idx
                items_scores += temp_items_scores
            temp_df = pd.DataFrame()
            temp_df['item'] = items_idx
            temp_df['score'] = items_scores
            temp_df = temp_df.sort_values(by='score', ascending=True)
            temp_df = temp_df.drop_duplicates(subset=['item'], keep='first', inplace=False)
            recall_item_list = temp_df['item'][:topk].values
            match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][recall_item_list])
        else:
            #普通召回
            items_idx, items_scores = annoy.query(v=user_emb, n=topk)  #the index of topk match items
            match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][items_idx])


    #get ground truth
    print("generate ground truth")
    # 额外加的，去重
    # for k in match_res:
    #     match_res[k] = np.unique(match_res[k]).tolist()[:200]
    # from collections import OrderedDict
    # for k in tqdm.tqdm(match_res):
    #     match_res[k] = list(OrderedDict.fromkeys(match_res[k]))[:200]

    data = pd.DataFrame({user_col: test_user[user_col], item_col: test_user[item_col]})
    data[user_col] = data[user_col].map(user_map)
    data[item_col] = data[item_col].map(item_map)
    user_pos_item = data.groupby(user_col).agg(list).reset_index()
    ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))  # user id -> ground truth

    print("compute topk metrics")
    if topk == 50:
        out = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=[50])
    elif topk == 20:
        out = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=[20])
    elif topk == 200:
        out = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=[50,100,200])
    print(out)
    return out


def get_item_sample_weight(items):
    #It is a effective weight used in word2vec
    items_cnt = Counter(items)
    p_sample = {item: count**0.75 for item, count in items_cnt.items()}
    p_sum = sum([v for k, v in p_sample.items()])
    item_sample_weight = {k: v / p_sum for k, v in p_sample.items()}
    return item_sample_weight
