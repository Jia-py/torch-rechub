import sys

sys.path.append("../..")
sys.path.append('/root/code/torch-rechub')

import os
# os.chdir("/root/code/torch-rechub")
import numpy as np
import pandas as pd
import torch
import copy
import tqdm
import nni

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch_rechub.models.matching import YoutubeDNN
from torch_rechub.models.matching.star import STAR
from torch_rechub.models.matching.daan import DAAN
from torch_rechub.models.matching.sar_net import SAR_NET
from torch_rechub.models.matching.m2m import M2M
from torch_rechub.models.matching.adi import ADI
from torch_rechub.models.matching.mmoe import MMOE
from torch_rechub.trainers import MatchTrainer
from torch_rechub.basic.features import DenseFeature, SparseFeature, SequenceFeature
from torch_rechub.utils.match import generate_seq_feature_match, gen_model_input
from torch_rechub.utils.data import df_to_dict, MatchDataGenerator
from movielens_utils import match_evaluation


def get_movielens_data(data_path, load_cache=False):
    data = pd.read_csv(data_path)
    data = data[data['tab'].isin([0, 1, 2])]
    data = data[data['is_click']==1]
    sparse_features = ['user_id', 'video_id', 'date', 'hourmin', 'is_click',
       'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate',
       'long_view', 'play_time_ms', 'duration_ms', 'profile_stay_time',
       'comment_stay_time', 'is_profile_enter', 'is_rand', 'tab',
       'user_active_degree', 'is_lowactive_period', 'is_live_streamer',
       'is_video_author', 'follow_user_num_range', 'fans_user_num_range',
       'friend_user_num_range', 'register_days_range', 'onehot_feat0',
       'onehot_feat1', 'onehot_feat2', 'onehot_feat3', 'onehot_feat4',
       'onehot_feat5', 'onehot_feat6', 'onehot_feat7', 'onehot_feat8',
       'onehot_feat9', 'onehot_feat10', 'onehot_feat11', 'onehot_feat12',
       'onehot_feat13', 'onehot_feat14', 'onehot_feat15', 'onehot_feat16',
       'onehot_feat17', 'video_type', 'upload_dt', 'upload_type',
       'visible_status', 'video_duration', 'server_width', 'server_height',
       'music_id', 'music_type', 'tag']
    user_col, item_col = 'user_id', 'video_id'

    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
        if feature == user_col:
            user_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}  #encode user id: raw user id
        if feature == item_col:
            item_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}  #encode item id: raw item id
    np.save("/root/code/torch-rechub/examples/matching/data/kuairand/saved/raw_id_maps.npy", np.array((user_map, item_map), dtype=object))

    # 用户和item固定的一些特征
    user_profile = data[['user_id', 'user_active_degree', 'is_lowactive_period',
       'is_live_streamer', 'is_video_author',
       'follow_user_num_range', 'fans_user_num_range',
        'friend_user_num_range',
       'register_days_range', 'onehot_feat0', 'onehot_feat1', 'onehot_feat2',
       'onehot_feat3', 'onehot_feat4', 'onehot_feat5', 'onehot_feat6',
       'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 'onehot_feat10',
       'onehot_feat11', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14',
       'onehot_feat15', 'onehot_feat16', 'onehot_feat17']].drop_duplicates(['user_id'])
    # user_profile = data[['user_id']].drop_duplicates(['user_id'])
    
    item_profile = data[['video_id', 'video_type', 'upload_dt', 'upload_type',
       'visible_status', 'video_duration', 'server_width', 'server_height',
       'music_id', 'music_type', 'tag']].drop_duplicates(['video_id'])
    
    neg_item_profile = item_profile.copy()
    # change the columns name in neg_item_profile
    neg_item_profile.columns = ['neg_items'] + ['neg_' + column for column in neg_item_profile.columns[1:]]
    # item_profile = data[['video_id']].drop_duplicates(['video_id'])

    if load_cache:  #if you have run this script before and saved the preprocessed data
        x_train, y_train, x_test = np.load("/root/code/torch-rechub/examples/matching/data/kuairand/saved/data_cache.npy", allow_pickle=True)
    else:
        #Note: mode=2 means list-wise negative sample generate, saved in last col "neg_items"
        df_train, df_test = generate_seq_feature_match(data,
                                                       data,
                                                       user_col,
                                                       item_col,
                                                    #    time_col='101',
                                                       cross_features=['date', 'hourmin',
                                                            'long_view', 'play_time_ms', 'duration_ms', 'profile_stay_time',
                                                            'comment_stay_time', 'is_profile_enter', 'is_rand', 'tab'],
                                                       sample_method=0,
                                                       mode=1,
                                                       neg_ratio=1,
                                                       min_item=0)
        # 把user profile和item profile拼上去, 这里因为item profile后面的信息不是随item_id唯一的，所以把item后面的特征放在cross_features里面了
        x_train = gen_model_input(df_train, user_profile, user_col, item_profile, neg_item_profile, item_col, seq_max_len=10)
        y_train = np.array([0] * df_train.shape[0])  #label=0 means the first pred value is positive sample
        x_test = gen_model_input(df_test, user_profile, user_col, item_profile, neg_item_profile, item_col, seq_max_len=10)
        np.save("/root/code/torch-rechub/examples/matching/data/kuairand/saved/data_cache.npy", np.array((x_train, y_train, x_test), dtype=object))

    # 双塔模型两边的特征
    user_cols = ['user_id', 'user_active_degree', 'is_lowactive_period',
       'is_live_streamer', 'is_video_author',
       'follow_user_num_range', 'fans_user_num_range',
        'friend_user_num_range',
       'register_days_range', 'onehot_feat0', 'onehot_feat1', 'onehot_feat2',
       'onehot_feat3', 'onehot_feat4', 'onehot_feat5', 'onehot_feat6',
       'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 'onehot_feat10',
       'onehot_feat11', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14',
       'onehot_feat15', 'onehot_feat16', 'onehot_feat17', 'date', 'hourmin',
        'long_view', 'play_time_ms', 'duration_ms', 'profile_stay_time',
        'comment_stay_time', 'is_profile_enter', 'is_rand', 'tab']
    # user_cols = ['user_id']
    item_cols = ['video_id', 'video_type', 'upload_dt', 'upload_type',
       'visible_status', 'video_duration', 'server_width', 'server_height',
       'music_id', 'music_type', 'tag']
    # item_cols = ['video_id']

    user_features = [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16) for name in user_cols]
    # user_features += [
    #     SequenceFeature("hist_video_id",
    #                     vocab_size=feature_max_idx['video_id'],
    #                     embed_dim=16,
    #                     pooling='mean',
    #                     shared_with='video_id')
    # ]

    item_features = [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16) for name in item_cols]
    neg_item_feature = [
        SparseFeature('neg_items', vocab_size=feature_max_idx['video_id'], embed_dim=16, shared_with='video_id')
    ] + [SparseFeature('neg_' + name, vocab_size=feature_max_idx[name], embed_dim=16, shared_with=name) for name in item_cols[1:]]

    # selected_features = ['121','122','124','125','127','206','207','210','216','508','509','702','853','301']
    # selected_features = ['121','122','124','125','127','128','129','301']
    selected_features = ['tab']
    selected_features = ['user_id', 'user_active_degree', 'is_lowactive_period',
       'is_live_streamer', 'is_video_author',
       'follow_user_num_range', 'fans_user_num_range',
        'friend_user_num_range',
       'register_days_range', 'onehot_feat0', 'onehot_feat1', 'onehot_feat2',
       'onehot_feat3', 'onehot_feat4', 'onehot_feat5', 'onehot_feat6',
       'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 'onehot_feat10',
       'onehot_feat11', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14',
       'onehot_feat15', 'onehot_feat16', 'onehot_feat17', 'hourmin',
        'long_view', 'play_time_ms', 'profile_stay_time',
        'comment_stay_time', 'is_profile_enter', 'is_rand', 'tab']
    selected_features = ['onehot_feat11','onehot_feat16','onehot_feat9','fans_user_num_range','onehot_feat6','onehot_feat7','is_rand','friend_user_num_range',
                         'user_id','onehot_feat4','tab','onehot_feat15','register_days_range','hourmin','onehot_feat3','onehot_feat13','onehot_feat0']
    DAU_input_features = [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16) for name in selected_features]

    all_item = df_to_dict(item_profile)
    test_user = x_test
    return user_features, item_features, DAU_input_features, neg_item_feature, x_train, y_train, all_item, test_user


def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # torch.manual_seed(seed)
    user_features, item_features, DAU_input_features, neg_item_feature, x_train, y_train, all_item, test_user = get_movielens_data(dataset_path, load_cache=True)
    dg = MatchDataGenerator(x=x_train, y=y_train)

    # train_user = set(x_train['101'].tolist())
    # test_user_lis = test_user['101'].tolist()
    # print('test user raw nums:', len(test_user_lis))
    # idxs = []
    # for i in tqdm.tqdm(range(len(test_user_lis))):
    #     if test_user_lis[i] in train_user:
    #         idxs.append(i)
    # test_user = {k: v[idxs] for k, v in test_user.items()}
    # print('test user nums:', len(idxs))

    if model_name == 'YoutubeDNN':
        model = YoutubeDNN(user_features, item_features, neg_item_feature, user_params={"dims": [16]}, temperature=0.02)
    elif model_name == 'STAR':
        model = STAR(user_features, item_features, neg_item_feature, user_params={"dims": [32, 16]})
    elif model_name == 'DAAN':
        model = DAAN(user_features, item_features, DAU_input_features, neg_item_feature)
    elif model_name == 'SAR_NET':
        model = SAR_NET(user_features, item_features, neg_item_feature)
    elif model_name == 'M2M':
        model = M2M(user_features, item_features, neg_item_feature, user_params={"dims": [ 16]})
    elif model_name == 'ADI':
        model = ADI(user_features, item_features, neg_item_feature, user_params={"dims": [ 16]})
    elif model_name == 'MMOE':
        model = MMOE(user_features, item_features, neg_item_feature)

    train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=batch_size)

    #mode=1 means pair-wise learning
    trainer = MatchTrainer(model,
                           mode=1,
                           optimizer_params={
                               "lr": learning_rate,
                               "weight_decay": weight_decay
                           },
                           n_epoch=epoch,
                           device=device,
                           model_path=save_dir,
                           test_dl = test_dl,
                           item_dl = item_dl,
                           test_user = test_user,
                           all_item = all_item,
                           model_name=model_name)

    
    trainer.fit(train_dl)

    model.load_state_dict(torch.load(os.path.join(save_dir, model_name, "model.pth")))
    print("inference embedding")
    user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=test_dl, model_path=save_dir)
    item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=save_dir)
    print(user_embedding.shape, item_embedding.shape)
    #torch.save(user_embedding.data.cpu(), save_dir + "user_embedding.pth")
    #torch.save(item_embedding.data.cpu(), save_dir + "item_embedding.pth")
    for slot_id in range(1,4):
        print('evaluation of slot_id: ', slot_id)
        idx_lis = np.where(test_user['tab']==slot_id)[0]
        # test_user_cp = copy.deepcopy(test_user)
        # for key in test_user_cp:
        #     test_user_cp[key] = test_user_cp[key][idx_lis]
        test_user_cp = {key: value[idx_lis] for key, value in test_user.items()}
        user_embedding_cp = user_embedding[idx_lis]
        match_evaluation(user_embedding_cp, item_embedding, test_user_cp, all_item, user_col='user_id', item_col='video_id', 
                         raw_id_maps='/root/code/torch-rechub/examples/matching/data/kuairand/saved/raw_id_maps.npy', topk=200)
    # match_evaluation(user_embedding, item_embedding, test_user, all_item, topk=100)
    # out = match_evaluation(user_embedding, item_embedding, test_user, all_item, user_col='user_id', item_col='video_id', 
    #                      raw_id_maps='/root/code/torch-rechub/examples/matching/data/kuairand/saved/raw_id_maps.npy', topk=200)
    
    # candidates = ['user_id', 'user_active_degree', 'is_lowactive_period',
    #    'is_live_streamer', 'is_video_author',
    #    'follow_user_num_range', 'fans_user_num_range',
    #     'friend_user_num_range',
    #    'register_days_range', 'onehot_feat0', 'onehot_feat1', 'onehot_feat2',
    #    'onehot_feat3', 'onehot_feat4', 'onehot_feat5', 'onehot_feat6',
    #    'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 'onehot_feat10',
    #    'onehot_feat11', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14',
    #    'onehot_feat15', 'onehot_feat16', 'onehot_feat17', 'hourmin',
    #     'long_view', 'play_time_ms', 'profile_stay_time',
    #     'comment_stay_time', 'is_profile_enter', 'is_rand', 'tab']
    # rank_lis = []
    # recall_50_raw, recall_100_raw, recall_200_raw = float(out['Recall'][0][11:]), float(out['Recall'][1][11:]), float(out['Recall'][2][11:])
    # avg_score_raw = (recall_50_raw + recall_100_raw + recall_200_raw) / 3
    # for idx, feature in enumerate(candidates):
    #     print('evaluation of feature: ', idx, feature)
    #     test_user_cp = test_user.copy()
    #     train_dl, test_dl, item_dl = dg.generate_dataloader(test_user_cp, all_item, batch_size=batch_size)
    #     user_embedding_cp = trainer.inference_embedding(model=model, mode="user", data_loader=test_dl, model_path=save_dir, mask_idx = idx)
    #     item_embedding_cp = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=save_dir)
    #     out = match_evaluation(user_embedding_cp, item_embedding_cp, test_user_cp, all_item, user_col='user_id', item_col='video_id',
    #                         raw_id_maps='/root/code/torch-rechub/examples/matching/data/kuairand/saved/raw_id_maps.npy', topk=200)
    #     recall_50, recall_100, recall_200 = float(out['Recall'][0][11:]), float(out['Recall'][1][11:]), float(out['Recall'][2][11:])
    #     rank_lis.append((feature, avg_score_raw - (recall_50+recall_100+recall_200)/3))
    # rank_lis = sorted(rank_lis, key=lambda x: x[1], reverse=True)
    # print(rank_lis)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/root/autodl-tmp/KuaiRand-pure-preprocessed.csv")
    parser.add_argument('--model_name', default='DAAN')
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2048)  #4096
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda:0')  #cuda:0
    parser.add_argument('--save_dir', default='/root/code/torch-rechub/examples/matching/data/kuairand/saved')
    parser.add_argument('--seed', type=int, default=2023)

    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device,
         args.save_dir, args.seed)
"""
python run_ml_youtube_dnn.py
"""