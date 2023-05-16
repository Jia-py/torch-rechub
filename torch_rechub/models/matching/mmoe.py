import torch
import torch.nn as nn
import torch.nn.functional as F
from ...basic.layers import MLP, EmbeddingLayer

class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        return out

class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        return out


class MMOE(torch.nn.Module):
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

        self.input_size = self.user_embedding_output_dims
        self.num_experts = 16
        self.experts_out = 64
        self.experts_hidden = 64
        self.towers_hidden = 64

        self.experts = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for _ in range(self.num_experts)])
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(self.input_size, self.num_experts), requires_grad=True) for i in range(3)])
        self.towers = nn.ModuleList([Tower(self.experts_out, 16, self.towers_hidden) for _ in range(3)])

        self.softmax = nn.Softmax(dim=1)

        self.linear1 = MLP(16, False, [ 16])

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
        
        experts_o = [e(input_user) for e in self.experts] # [b, experts_out]
        experts_o_tensor = torch.stack(experts_o) # num_experts, b, experts_out

        gates_o = [self.softmax(torch.matmul(input_user, w)) for w in self.w_gates] # [b, num_experts]

        tower_input = [g.t().unsqueeze(2).expand(-1,-1,self.experts_out) * experts_o_tensor for g in gates_o] # [num_experts, b, experts_out]
        tower_input = [torch.sum(t, dim=0) for t in tower_input] # [b, experts_out]

        final_output = [t(ti) for t,ti in zip(self.towers, tower_input)] # [3, b, 16]

        user_embedding = torch.zeros(input_user.shape[0], 16).to(input_user.device)
        user_embedding = torch.where(slot1_mask.unsqueeze(1), final_output[0], user_embedding)
        user_embedding = torch.where(slot2_mask.unsqueeze(1), final_output[1], user_embedding)
        user_embedding = torch.where(slot3_mask.unsqueeze(1), final_output[2], user_embedding)

        user_embedding = user_embedding.reshape(user_embedding.shape[0],1,16)

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