import torch
import torch.nn as nn
import torch.nn.functional as F
from ...basic.layers import MLP, EmbeddingLayer

class debias_expert_net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, 16)

    def forward(self, x):
        x = self.bn(x)
        x = self.linear(x)
        return x


class SAR_NET(torch.nn.Module):
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

    def __init__(self, user_features, item_features, neg_item_feature):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.neg_item_feature = neg_item_feature
        self.user_dims = sum([fea.embed_dim for fea in user_features])
        self.embedding = EmbeddingLayer(user_features + item_features)
        self.mode = None

        self.user_embedding_output_dims = len(self.user_features) * 16
        self.auxiliary = MLP(self.user_embedding_output_dims, False, [64,32,16], 0)

        self.shared_weight = nn.Parameter(torch.empty(self.user_embedding_output_dims, 64))
        self.shared_bias = nn.Parameter(torch.zeros(64))

        self.slot1_weight = nn.Parameter(torch.empty(self.user_embedding_output_dims, 64))
        self.slot1_bias = nn.Parameter(torch.zeros(64))
        self.slot2_weight = nn.Parameter(torch.empty(self.user_embedding_output_dims, 64))
        self.slot2_bias = nn.Parameter(torch.zeros(64))
        self.slot3_weight = nn.Parameter(torch.empty(self.user_embedding_output_dims, 64))
        self.slot3_bias = nn.Parameter(torch.zeros(64))

        for m in [self.shared_weight, self.slot1_weight, self.slot2_weight, self.slot3_weight]:
            torch.nn.init.xavier_uniform_(m.data)

        self.linear1 = MLP(5*16, False, [64, 32, 16])
        self.linear2 = torch.nn.Linear(64, 10)

        self.shared_expert = nn.ModuleList([debias_expert_net() for _ in range(8)])
        self.specific_expert = nn.ModuleList([debias_expert_net() for _ in range(6)])

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
        slot_id = x['301']
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)  #[batch_size, num_features*deep_dims]

        slot1_mask = (slot_id == 1)
        slot2_mask = (slot_id == 2)
        slot3_mask = (slot_id == 3)
        slot1_output = torch.matmul(input_user, torch.multiply(self.slot1_weight, self.shared_weight))+self.slot1_bias+self.shared_bias
        slot2_output = torch.matmul(input_user, torch.multiply(self.slot2_weight, self.shared_weight))+self.slot2_bias+self.shared_bias
        slot3_output = torch.matmul(input_user, torch.multiply(self.slot3_weight, self.shared_weight))+self.slot3_bias+self.shared_bias
        output = torch.zeros_like(slot1_output)
        output = torch.where(slot1_mask.unsqueeze(1), slot1_output, output)
        output = torch.where(slot2_mask.unsqueeze(1), slot2_output, output)
        output = torch.where(slot3_mask.unsqueeze(1), slot3_output, output) # b,64

        shared_output = torch.stack([self.shared_expert[i](output) for i in range(8)], dim=1) # b,8,16
        slot1_output = torch.stack([self.specific_expert[i](output) for i in [0,1]], dim=1) # b,2,16
        slot2_output = torch.stack([self.specific_expert[i](output) for i in [2,3]], dim=1)
        slot3_output = torch.stack([self.specific_expert[i](output) for i in [4,5]], dim=1)
        expert_specific_output = torch.zeros((output.shape[0], 2, 16)).to(output.device)
        expert_specific_output = torch.where(slot1_mask.unsqueeze(1).unsqueeze(2), slot1_output, expert_specific_output)
        expert_specific_output = torch.where(slot2_mask.unsqueeze(1).unsqueeze(2), slot2_output, expert_specific_output)
        expert_specific_output = torch.where(slot3_mask.unsqueeze(1).unsqueeze(2), slot3_output, expert_specific_output)
        expert_output = torch.concat([shared_output, expert_specific_output], dim=1) # b,10,16

        gate = torch.softmax(self.linear2(output), dim=-1) # b,10
        expert_output = torch.mul(expert_output, gate.unsqueeze(2)) # b,10,16
        user_embedding = expert_output.sum(dim=1).unsqueeze(1) # b,1,16

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
                                        squeeze_dim=False).squeeze(1)  #[batch_size, n_neg_items, embed_dim]
        neg_embeddings = self.linear1(neg_embeddings.reshape(neg_embeddings.shape[0],-1)).reshape(neg_embeddings.shape[0],1,16)
        neg_embeddings = F.normalize(neg_embeddings, p=2, dim=2)
        return torch.cat((pos_embedding, neg_embeddings), dim=1)  #[batch_size, 1+n_neg_items, embed_dim]