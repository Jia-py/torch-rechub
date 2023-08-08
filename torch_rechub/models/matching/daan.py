import torch
import torch.nn as nn
import torch.nn.functional as F
from ...basic.layers import MLP, EmbeddingLayer

class DAU(nn.Module):
    def __init__(self, feature_num, embed_size, hid_dims, output_dims):
        super().__init__()
        '''
        hid_dims: some hidden layer dims
        output_dims: dims to fit other modules in model. e.g. [f*e, output_dim]. It's also the requested shape of output
        '''
        self.output_dims = output_dims
        self.linear_layers = nn.ModuleList([MLP(embed_size, False, hid_dims) for i in range(feature_num)])
        self.shared_weight = nn.Parameter(torch.empty(output_dims[0], output_dims[1]))
        self.shared_bias = nn.Parameter(torch.zeros(output_dims[1]))
        torch.nn.init.xavier_uniform_(self.shared_weight.data)
        self.trans = MLP(feature_num * hid_dims[-1], False, [feature_num * hid_dims[-1]], 0)
        self.trans_weight = MLP(feature_num * hid_dims[-1], False, [output_dims[0]*output_dims[1]], 0)
        self.trans_bias = MLP(feature_num * hid_dims[-1], False, [output_dims[1]], 0)


    def forward(self, x):
        b, f, e = x.shape
        trans_features = []
        for i in range(f):
            feature = x[:, i, :].clone().detach()
            feature = self.linear_layers[i](feature)
            trans_features.append(feature)
        trans_features = torch.stack(trans_features, dim=1)
        residual_output = self.trans(trans_features.view(b,-1)) + trans_features.reshape(b, -1) # b, f*hid[0]
        # residual_output = self.trans(trans_features.view(b,-1))
        specific_weight = self.trans_weight(residual_output).reshape(b, self.output_dims[0], self.output_dims[1])
        weight = torch.multiply(specific_weight, self.shared_weight.unsqueeze(0))
        specific_bias = self.trans_bias(residual_output).reshape(b, self.output_dims[1])
        bias = specific_bias + self.shared_bias.unsqueeze(0)
        return weight, bias

class DAAN(torch.nn.Module):

    def __init__(self, user_features, item_features, DAU_input_feature, neg_item_feature):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.neg_item_feature = neg_item_feature
        self.user_dims = sum([fea.embed_dim for fea in user_features])
        self.embedding = EmbeddingLayer(user_features + item_features)
        self.mode = None

        self.DAU_input_feature = DAU_input_feature

        self.user_embedding_output_dims = len(self.user_features) * 16
        self.DAU_fs = DAU(len(self.DAU_input_feature), 16, [8,16], [len(self.user_features), len(self.user_features)])
        self.DAU_tr = DAU(len(self.DAU_input_feature), 16,[8,16], [self.user_embedding_output_dims, 64])

        self.output_mlp = MLP(64, False, [16])

        self.linear1 = MLP(11 * 16, False, [16]) # item_cols * 16
        
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
        DAU_input = self.embedding(x, self.DAU_input_feature, squeeze_dim=True).reshape(input_user.shape[0], len(self.DAU_input_feature), 16)
        b,f,e = input_user.shape[0], len(self.user_features), 16 
        input_user = input_user.reshape(b,f,e)

        fs_weight, fs_bias = self.DAU_fs(DAU_input)
        se_weight = torch.matmul(torch.mean(input_user, dim=2, keepdim=True).reshape(b,1,-1), fs_weight).reshape(b,-1) + fs_bias
        se_output = torch.multiply(input_user, se_weight.unsqueeze(2))

        tr_weight, tr_bias = self.DAU_tr(DAU_input)
        tr_output = torch.matmul(se_output.reshape(b,1,-1), tr_weight).reshape(b,-1) + tr_bias

        user_embedding = tr_output.reshape(b, 1, -1)

        user_embedding = self.output_mlp(tr_output).reshape(b,1,-1)  #[batch_size, 1, embed_dim]
        user_embedding = F.normalize(user_embedding, p=2, dim=2)
        if self.mode == "user":
            return user_embedding.squeeze(1)  #inference embedding mode -> [batch_size, embed_dim]
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        pos_embedding = self.embedding(x, self.item_features, squeeze_dim=False)  #[batch_size, 1, embed_dim]
        b = pos_embedding.shape[0]
        pos_embedding = self.linear1(pos_embedding.reshape(b,-1)).reshape(b, 1,-1)
        pos_embedding = F.normalize(pos_embedding, p=2, dim=2)
        if self.mode == "item":  #inference embedding mode
            return pos_embedding.reshape(pos_embedding.shape[0],-1)  #[batch_size, embed_dim]
        neg_embeddings = self.embedding(x, self.neg_item_feature, squeeze_dim=False)  #[batch_size, n_neg_items, embed_dim]
        neg_embeddings = self.linear1(neg_embeddings.reshape(b,-1)).reshape(b,-1,16)
        neg_embeddings = F.normalize(neg_embeddings, p=2, dim=2)
        return torch.cat((pos_embedding, neg_embeddings), dim=1)  #[batch_size, 1+n_neg_items, embed_dim]
