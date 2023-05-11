"""
Date: create on 23/05/2022
References: 
    paper: (RecSys'2016) Deep Neural Networks for YouTube Recommendations
    url: https://dl.acm.org/doi/10.1145/2959100.2959190
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch
import torch.nn.functional as F
from ...basic.layers import MLP, EmbeddingLayer


class YoutubeDNN(torch.nn.Module):
    """The match model mentioned in `Deep Neural Networks for YouTube Recommendations` paper.
    It's a DSSM match model trained by global softmax loss on list-wise samples.
    Note in origin paper, it's without item dnn tower and train item embedding directly.

    Args:
        user_features (list[Feature Class]): training by the user tower module.
        item_features (list[Feature Class]): training by the embedding table, it's the item id feature.
        neg_item_feature (list[Feature Class]): training by the embedding table, it's the negative items id feature.
        user_params (dict): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        temperature (float): temperature factor for similarity score, default to 1.0.
    """

    def __init__(self, user_features, item_features, neg_item_feature, user_params, temperature=1.0):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.neg_item_feature = neg_item_feature
        self.temperature = temperature
        self.user_dims = sum([fea.embed_dim for fea in user_features])
        self.embedding = EmbeddingLayer(user_features + item_features)
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        self.mode = None

        self.linear1 = MLP(3*16, False, [64, 32, 16])

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding

        # calculate cosine score
        pos_embedding = item_embedding[:, 0, :]
        neg_embedding = item_embedding[:, 1, :]
        pos_score = torch.mul(user_embedding.squeeze(1), pos_embedding).sum(dim=1)
        neg_score = torch.mul(user_embedding.squeeze(1), neg_embedding).sum(dim=1)
        return pos_score, neg_score

    def user_tower(self, x):
        if self.mode == "item":
            return None
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)  #[batch_size, num_features*deep_dims]
        user_embedding = self.user_mlp(input_user).unsqueeze(1)  #[batch_size, 1, embed_dim]
        user_embedding = F.normalize(user_embedding, p=2, dim=2)
        if self.mode == "user":
            return user_embedding.squeeze(1)  #inference embedding mode -> [batch_size, embed_dim]
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        pos_embedding = self.embedding(x, self.item_features, squeeze_dim=False)  #[batch_size, 1, embed_dim]
        pos_embedding = self.linear1(pos_embedding.reshape(pos_embedding.shape[0],-1)).reshape(pos_embedding.shape[0],1,16)
        pos_embedding = F.normalize(pos_embedding, p=2, dim=2)
        if self.mode == "item":  #inference embedding mode
            return pos_embedding.reshape(pos_embedding.shape[0],-1)  #[batch_size, embed_dim]
        neg_embeddings = self.embedding(x, self.neg_item_feature,
                                        squeeze_dim=False)  #[batch_size, n_neg_items, embed_dim]
        neg_embeddings = self.linear1(neg_embeddings.reshape(neg_embeddings.shape[0],-1)).reshape(neg_embeddings.shape[0],1,16)
        neg_embeddings = F.normalize(neg_embeddings, p=2, dim=2)
        return torch.cat((pos_embedding, neg_embeddings), dim=1)  #[batch_size, 1+n_neg_items, embed_dim]
