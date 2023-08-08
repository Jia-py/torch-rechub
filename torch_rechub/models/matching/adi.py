import torch
import torch.nn as nn
import torch.nn.functional as F
from ...basic.layers import MLP, EmbeddingLayer


class ADI(torch.nn.Module):
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

    def __init__(self, user_features, item_features, neg_item_feature, user_params):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.neg_item_feature = neg_item_feature
        self.user_dims = sum([fea.embed_dim for fea in user_features])
        self.embedding = EmbeddingLayer(user_features + item_features)
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        self.mode = None

        self.user_embedding_output_dims = len(self.user_features) * 16

        self.shared_mlp = MLP(self.user_embedding_output_dims, output_layer=False, dims=[64, 32])
        self.slot1_mlp = MLP(self.user_embedding_output_dims, output_layer=False, dims=[64, 32])
        self.slot2_mlp = MLP(self.user_embedding_output_dims, output_layer=False, dims=[64, 32])
        self.slot3_mlp = MLP(self.user_embedding_output_dims, output_layer=False, dims=[64, 32])

        self.fusion_slot1_specific_W = nn.Parameter(torch.empty(16, 1))
        self.fusion_slot2_specific_W = nn.Parameter(torch.empty(16, 1))
        self.fusion_slot3_specific_W = nn.Parameter(torch.empty(16, 1))
        self.fusion_slot1_shared_W = nn.Parameter(torch.empty(16, 1))
        self.fusion_slot2_shared_W = nn.Parameter(torch.empty(16, 1))
        self.fusion_slot3_shared_W = nn.Parameter(torch.empty(16, 1))

        # ------------------------------------------------------------------------------

        self.shared_weight = nn.Parameter(torch.empty(3*32, 32))
        self.shared_bias = nn.Parameter(torch.zeros(32))

        self.slot1_weight = nn.Parameter(torch.empty(3*32, 32))
        self.slot1_bias = nn.Parameter(torch.zeros(32))
        self.slot2_weight = nn.Parameter(torch.empty(3*32, 32))
        self.slot2_bias = nn.Parameter(torch.zeros(32))
        self.slot3_weight = nn.Parameter(torch.empty(3*32, 32))
        self.slot3_bias = nn.Parameter(torch.zeros(32))


        self.mlp = MLP(32, False, [64, 16], 0)

        self.se_weight1 = nn.Parameter(torch.empty(len(self.user_features), len(self.user_features)))
        self.se_weight2 = nn.Parameter(torch.empty(len(self.user_features), len(self.user_features)))
        self.se_weight3 = nn.Parameter(torch.empty(len(self.user_features), len(self.user_features)))

        for m in [self.shared_weight, self.slot1_weight, self.slot2_weight, self.slot3_weight, self.se_weight1, self.se_weight2, self.se_weight3, self.fusion_slot1_specific_W, self.fusion_slot2_specific_W, self.fusion_slot3_specific_W, self.fusion_slot1_shared_W, self.fusion_slot2_shared_W, self.fusion_slot3_shared_W]:
            torch.nn.init.xavier_uniform_(m.data)

        self.linear1 = MLP(11 * 16, False, [16])

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
        slot_id = x['tab']
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)  #[batch_size, num_features*deep_dims]
        slot = input_user.reshape(input_user.shape[0], len(self.user_features), 16)[:,-1,:] # b,16

        slot1_mask = (slot_id == 1)
        slot2_mask = (slot_id == 2)
        slot3_mask = (slot_id == 3)

        b = input_user.shape[0]

        se_weight_1 = torch.matmul(torch.mean(input_user.reshape(input_user.shape[0], len(self.user_features), 16), dim=2).unsqueeze(1), self.se_weight1).reshape(b, -1) # b, 1, f
        se_weight_2 = torch.matmul(torch.mean(input_user.reshape(input_user.shape[0], len(self.user_features), 16), dim=2).unsqueeze(1), self.se_weight2).reshape(b, -1) # b, 1, f
        se_weight_3 = torch.matmul(torch.mean(input_user.reshape(input_user.shape[0], len(self.user_features), 16), dim=2).unsqueeze(1), self.se_weight3).reshape(b, -1) # b, 1, f
        se_weight = torch.zeros_like(se_weight_1)
        se_weight = torch.where(slot1_mask.unsqueeze(1), se_weight_1, se_weight)
        se_weight = torch.where(slot2_mask.unsqueeze(1), se_weight_2, se_weight)
        se_weight = torch.where(slot3_mask.unsqueeze(1), se_weight_3, se_weight)
        input_user = torch.mul(input_user.reshape(b, len(self.user_features), -1), se_weight.unsqueeze(2))
        input_user = input_user.reshape(b, -1) # b, f*e

        shared_output = self.shared_mlp(input_user)
        slot1_output = self.slot1_mlp(input_user)
        slot2_output = self.slot2_mlp(input_user)
        slot3_output = self.slot3_mlp(input_user)
        specific_output = torch.zeros_like(slot1_output)
        specific_output = torch.where(slot1_mask.unsqueeze(1), slot1_output, specific_output)
        specific_output = torch.where(slot2_mask.unsqueeze(1), slot2_output, specific_output)
        specific_output = torch.where(slot3_mask.unsqueeze(1), slot3_output, specific_output)  # b,32

        slot1_specific_beta = torch.matmul(slot, self.fusion_slot1_specific_W) # b, 1
        slot2_specific_beta = torch.matmul(slot, self.fusion_slot2_specific_W) # b, 1
        slot3_specific_beta = torch.matmul(slot, self.fusion_slot3_specific_W) # b, 1
        slot1_shared_beta = torch.matmul(slot, self.fusion_slot1_shared_W) # b, 1
        slot2_shared_beta = torch.matmul(slot, self.fusion_slot2_shared_W) # b, 1
        slot3_shared_beta = torch.matmul(slot, self.fusion_slot3_shared_W) # b, 1
        specific_beta = torch.zeros_like(slot1_specific_beta)
        specific_beta = torch.where(slot1_mask.unsqueeze(1), slot1_specific_beta, specific_beta)
        specific_beta = torch.where(slot2_mask.unsqueeze(1), slot2_specific_beta, specific_beta)
        specific_beta = torch.where(slot3_mask.unsqueeze(1), slot3_specific_beta, specific_beta)
        shared_beta = torch.zeros_like(slot1_shared_beta)
        shared_beta = torch.where(slot1_mask.unsqueeze(1), slot1_shared_beta, shared_beta)
        shared_beta = torch.where(slot2_mask.unsqueeze(1), slot2_shared_beta, shared_beta)
        shared_beta = torch.where(slot3_mask.unsqueeze(1), slot3_shared_beta, shared_beta)

        fusion_output = torch.concat([specific_output * specific_beta, torch.mul(shared_output * shared_beta, specific_output * specific_beta), shared_output * shared_beta], dim=1) # b, 3*32


        slot1_output = torch.matmul(fusion_output, torch.multiply(self.slot1_weight, self.shared_weight))+self.slot1_bias+self.shared_bias
        slot2_output = torch.matmul(fusion_output, torch.multiply(self.slot2_weight, self.shared_weight))+self.slot2_bias+self.shared_bias
        slot3_output = torch.matmul(fusion_output, torch.multiply(self.slot3_weight, self.shared_weight))+self.slot3_bias+self.shared_bias
        output = torch.zeros_like(slot1_output)
        output = torch.where(slot1_mask.unsqueeze(1), slot1_output, output)
        output = torch.where(slot2_mask.unsqueeze(1), slot2_output, output)
        output = torch.where(slot3_mask.unsqueeze(1), slot3_output, output)
        output = self.mlp(output) # b,16
        user_embedding = (output).unsqueeze(1) # b,1,16
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