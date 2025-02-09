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
    data_type = {'click':np.int8, 'purchase': np.int8, '101':np.int32, '121':np.uint8, '122':np.uint8, '124':np.uint8, '125':np.uint8, '126':np.uint8, '127':np.uint8, '128':np.uint8, '129':np.uint8, '205':np.int32, '206':np.int16, '207':np.int32, '210':np.int32, '216':np.int32, '508':np.int16, '509':np.int32, '702':np.int32, '853':np.int32, '301':np.int8, '109_14':np.int16, '110_14':np.int32, '127_14':np.int32, '150_14':np.int32, 'D109_14': np.float16, 'D110_14': np.float16, 'D127_14': np.float16, 'D150_14': np.float16, 'D508': np.float16, 'D509': np.float16, 'D702': np.float16, 'D853': np.float16}
    data = pd.read_csv(data_path, dtype=data_type)
    sparse_features = ['101', '121','122','124','125','126','127','128','129','205','206','207','210','216','508','509','702','853','301','109_14','110_14','127_14','150_14']
    user_col, item_col = "101", "205"
    pos_data = data[data['click'] == 1].reset_index(drop=True)

    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        pos_data[feature] = lbe.transform(pos_data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
        if feature == user_col:
            user_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}  #encode user id: raw user id
        if feature == item_col:
            item_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}  #encode item id: raw item id
    np.save("/root/code/torch-rechub/examples/ranking/data/ali-ccp/saved/raw_id_maps.npy", np.array((user_map, item_map), dtype=object))

    # 用户和item固定的一些特征
    user_profile = data[["101","121","122","124","125","126","127","128","129"]].drop_duplicates(['101'])
    # item_profile = data[["205", "207", "216"]].drop_duplicates(['205'])
    item_profile = data[["205"]].drop_duplicates()
    # item_profile = data[["205","206","207","210","216"]].drop_duplicates()

    if load_cache:  #if you have run this script before and saved the preprocessed data
        x_train, y_train, x_test = np.load("/root/code/torch-rechub/examples/ranking/data/ali-ccp/saved/data_cache.npy", allow_pickle=True)
    else:
        #Note: mode=2 means list-wise negative sample generate, saved in last col "neg_items"
        df_train, df_test = generate_seq_feature_match( pos_data,
                                                       data,
                                                       user_col,
                                                       item_col,
                                                    #    time_col='101',
                                                       cross_features=['301',"206","207","210","216"],
                                                       sample_method=0,
                                                       mode=1,
                                                       neg_ratio=6,
                                                       min_item=0)
        # 把user profile和item profile拼上去, 这里因为item profile后面的信息不是随item_id唯一的，所以把item后面的特征放在cross_features里面了
        x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=5)
        y_train = np.array([0] * df_train.shape[0])  #label=0 means the first pred value is positive sample
        x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=5)
        np.save("/root/code/torch-rechub/examples/ranking/data/ali-ccp/saved/data_cache.npy", np.array((x_train, y_train, x_test), dtype=object))

    # 双塔模型两边的特征
    user_cols = ["101","121","122","124","125","126","127","128","129",'301']
    # user_cols = ["101"]
    # user_cols = ["101","121","122","124","125","126","127","128","129"]
    # item_cols = ['205',"206","207","210","216"]
    # item_cols = ['205', "207", "216"]
    item_cols = ['205']

    user_features = [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16) for name in user_cols]
    # user_features += [
    #     SequenceFeature("hist_205",
    #                     vocab_size=feature_max_idx['205'],
    #                     embed_dim=16,
    #                     pooling='mean', 
    #                     shared_with='205')
    # ]

    item_features = [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16) for name in item_cols]
    neg_item_feature = [
        SparseFeature('neg_items', vocab_size=feature_max_idx['205'], embed_dim=16, shared_with='205')
    ] + [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16) for name in item_cols[1:]]

    # selected_features = ['121','122','124','125','127','206','207','210','216','508','509','702','853','301']
    selected_features = ['122','124','125','127','129','301']
    # selected_features = ['301']
    DAU_input_features = [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16, shared_with=name) for name in selected_features]

    all_item = df_to_dict(item_profile)
    test_user = x_test
    # 只选1w个用户
    # for key in test_user:
    #     test_user[key] = test_user[key][:10000]
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
        model = YoutubeDNN(user_features, item_features, neg_item_feature, user_params={"dims": [64, 32, 16]}, temperature=0.02)
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
                           test_dl=test_dl,
                           item_dl=item_dl,
                           test_user = test_user,
                           all_item = all_item,
                           model_name = model_name)

    
    trainer.fit(train_dl)

    print("inference embedding")
    model.load_state_dict(torch.load(os.path.join(save_dir, model_name, "model.pth")))
    user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=test_dl, model_path=save_dir)
    item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=save_dir)
    print(user_embedding.shape, item_embedding.shape)
    #torch.save(user_embedding.data.cpu(), save_dir + "user_embedding.pth")
    #torch.save(item_embedding.data.cpu(), save_dir + "item_embedding.pth")
    for slot_id in range(1,4):
        print('evaluation of slot_id: ', slot_id)
        idx_lis = np.where(test_user['301']==slot_id)[0]
        # test_user_cp = copy.deepcopy(test_user)
        # for key in test_user_cp:
        #     test_user_cp[key] = test_user_cp[key][idx_lis]
        test_user_cp = {key: value[idx_lis] for key, value in test_user.items()}
        user_embedding_cp = user_embedding[idx_lis]
        # match_evaluation(user_embedding_cp, item_embedding, test_user_cp, all_item, topk=50)
        # match_evaluation(user_embedding_cp, item_embedding, test_user_cp, all_item, topk=100)
        out = match_evaluation(user_embedding_cp, item_embedding, test_user_cp, all_item, user_col='101', item_col='205', topk=200)
    # match_evaluation(user_embedding, item_embedding, test_user, all_item, topk=100)
    # match_evaluation(user_embedding, item_embedding, test_user, all_item, user_col='101', item_col='205', topk=200)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/root/autodl-tmp/ali_ccp_train.csv")
    parser.add_argument('--model_name', default='DAAN')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4096)  #4096
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0')  #cuda:0
    parser.add_argument('--save_dir', default='/root/code/torch-rechub/examples/matching/data/ali-ccp/saved')
    parser.add_argument('--seed', type=int, default=2023)

    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device,
         args.save_dir, args.seed)
"""
python run_ml_youtube_dnn.py
"""