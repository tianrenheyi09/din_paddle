import torch.nn as nn
import torch


import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, feature_dim, embedding_dim):
        super().__init__()

        self.embed = nn.Embedding(feature_dim, embedding_dim, padding_idx=0)

        # normal weight initialization
        self.embed.weight.data.normal_(0., 0.0001)
        # TODO: regularization

    def forward(self, x):
        return self.embed(x)



class Dice(nn.Module):
    def __init__(self, num_features, dim=2):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(num_features, eps=1e-9)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        if self.dim == 3:
            self.alpha = torch.zeros((num_features, 1)).cuda()
        elif self.dim == 2:
            self.alpha = torch.zeros((num_features,)).cuda()

    def forward(self, x):
        if self.dim == 3:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)

        elif self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x

        return out




# from dice import Dice

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias, batch_norm=True, dropout_rate=0.5, activation='relu',
                 sigmoid=False, dice_dim=2):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_size) >= 1 and len(bias) >= 1
        assert len(bias) == len(hidden_size)
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0], bias=bias[0]))

        for i, h in enumerate(hidden_size[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size[i]))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'dice':
                assert dice_dim
                layers.append(Dice(hidden_size[i], dim=dice_dim))
            elif activation.lower() == 'prelu':
                layers.append(nn.PReLU())
            else:
                raise NotImplementedError

            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1], bias=bias[i]))

        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()

        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)



class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim=4):
        super(AttentionSequencePoolingLayer, self).__init__()

        # TODO: DICE acitivation function
        # TODO: attention weight normalization
        self.local_att = LocalActivationUnit(hidden_size=[64, 16], bias=[True, True], embedding_dim=embedding_dim,
                                             batch_norm=False)

    def forward(self, query_ad, user_behavior, user_behavior_length):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # user behavior length: size -> batch_size * 1
        # output              : size -> batch_size * 1 * embedding_size

        attention_score = self.local_att(query_ad, user_behavior)
        attention_score = torch.transpose(attention_score, 1, 2)  # B * 1 * T
        # print(attention_score.size())

        # define mask by length
        user_behavior_length = user_behavior_length.type(torch.LongTensor)
        mask = torch.arange(user_behavior.size(1))[None, :] < user_behavior_length[:, None]

        # mask
        output = torch.mul(attention_score, mask.type(torch.cuda.FloatTensor))  # batch_size *

        # multiply weight
        output = torch.matmul(output, user_behavior)

        return output


class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_size=[80, 40], bias=[True, True], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4 * embedding_dim,
                                       hidden_size=hidden_size,
                                       bias=bias,
                                       batch_norm=batch_norm,
                                       activation='dice',
                                       dice_dim=3)

        self.fc2 = FullyConnectedLayer(input_size=hidden_size[-1],
                                       hidden_size=[1],
                                       bias=[True],
                                       batch_norm=batch_norm,
                                       activation='dice',
                                       dice_dim=3)
        # TODO: fc_2 initialization

    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(1)
        queries = torch.cat([query for _ in range(user_behavior_len)], dim=1)

        attention_input = torch.cat([queries, user_behavior, queries - user_behavior, queries * user_behavior], dim=-1)
        attention_output = self.fc1(attention_input)
        attention_output = self.fc2(attention_output)

        return attention_output




# from .embedding import EmbeddingLayer
# from .fc import FullyConnectedLayer
# from .attention import AttentionSequencePoolingLayer

dim_config = {
    'user_exposed_time': 24,
    'user_gender': 2,
    'user_age': 9,
    'history_article_id': 53932,  # multi-hot
    'history_image_feature': 2048,
    'history_categories': 23,
    'query_article_id': 1856,  # one-hot
    'query_image_feature': 2048,
    'query_categories': 23
}

que_embed_features = ['query_article_id']
que_image_features = ['query_image_feature']
que_category = ['query_categories']

his_embed_features = ['history_article_id']
his_image_features = ['history_image_feature']
his_category = ['history_categories']

image_hidden_dim = 64
category_dim = 23

embed_features = [k for k, _ in dim_config.items() if 'user' in k]


class DeepInterestNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embedding_size = config['embedding_size']

        self.query_feature_embedding_dict = dict()
        for feature in que_embed_features:
            self.query_feature_embedding_dict[feature] = EmbeddingLayer(feature_dim=dim_config[feature],
                                                                        embedding_dim=embedding_size).cuda()
        self.query_image_fc = FullyConnectedLayer(input_size=2048,
                                                  hidden_size=[image_hidden_dim],
                                                  bias=[True],
                                                  activation='relu').cuda()

        self.history_feature_embedding_dict = dict()
        for feature in his_embed_features:
            self.history_feature_embedding_dict[feature] = EmbeddingLayer(feature_dim=dim_config[feature],
                                                                          embedding_dim=embedding_size).cuda()
        self.history_image_fc = FullyConnectedLayer(input_size=2048,
                                                    hidden_size=[image_hidden_dim],
                                                    bias=[True],
                                                    activation='relu').cuda()

        self.attn = AttentionSequencePoolingLayer(embedding_dim=image_hidden_dim + embedding_size + category_dim).cuda()
        self.fc_layer = FullyConnectedLayer(input_size=2 * (image_hidden_dim + embedding_size + category_dim) + sum(
            [dim_config[k] for k in embed_features]),
                                            hidden_size=[200, 80, 1],
                                            bias=[True, True, False],
                                            activation='relu',
                                            sigmoid=True).cuda()

    def forward(self, user_features):
        # user_features -> dict (key:feature name, value: feature tensor)

        # deep input embedding
        feature_embedded = []

        for feature in embed_features:
            feature_embedded.append(user_features[feature])

        feature_embedded = torch.cat(feature_embedded, dim=1)
        # print('User_feature_embed size', user_feature_embedded.size()) # batch_size * (feature_size * embedding_size)
        # print('User feature done')

        query_feature_embedded = []

        for feature in que_embed_features:
            query_feature_embedded.append(self.query_feature_embedding_dict[feature](user_features[feature].squeeze()))
        for feature in que_image_features:
            query_feature_embedded.append(self.query_image_fc(user_features[feature]))
        for feature in que_category:
            query_feature_embedded.append(user_features[feature])

        query_feature_embedded = torch.cat(query_feature_embedded, dim=1)
        # print('Query feature_embed size', query_feature_embedded.size()) # batch_size * (feature_size * embedding_size)
        # print('Query feature done')
        # exit()

        # TODO: history
        history_feature_embedded = []
        for feature in his_embed_features:
            # print(feature)
            # print(user_features[feature].size())
            history_feature_embedded.append(self.history_feature_embedding_dict[feature](user_features[feature]))
            # print(self.history_feature_embedding_dict[feature](user_features[feature]).size())

        for feature in his_image_features:
            # print(user_features[feature].size())
            history_feature_embedded.append(self.history_image_fc(user_features[feature]))
        for feature in his_category:
            history_feature_embedded.append(user_features[feature])

        history_feature_embedded = torch.cat(history_feature_embedded, dim=2)
        # print('History feature_embed size', history_feature_embedded.size()) # batch_size * T * (feature_size * embedding_size)
        # print('History feature done')

        # print(user_features['history_len'])
        # print(user_features['history_len'].size())

        history = self.attn(query_feature_embedded.unsqueeze(1),
                            history_feature_embedded,
                            user_features['history_len'])

        concat_feature = torch.cat([feature_embedded, query_feature_embedded, history.squeeze()], dim=1)

        # fully-connected layers
        # print(concat_feature.size())
        output = self.fc_layer(concat_feature)
        return output


if __name__ == "__main__":
    a = DeepInterestNetwork()
    import torch
    import numpy as np

    user_feature = {
        'user_exposed_time': torch.LongTensor(np.zeros(shape=(2, 24))),
        'user_gender': torch.LongTensor(np.zeros(shape=(2, 2))),
        'user_age': torch.LongTensor(np.zeros(shape=(2, 9))),
    }
    a(user_feature)