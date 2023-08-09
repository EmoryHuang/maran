import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask=None, other=None):
        # Q shape: (batch_size, n_heads, len_q, d_k)
        # K shape: (batch_size, n_heads, len_k, d_k)
        # V shape: (batch_size, n_heads, len_v(=len_k), d_v)
        # attn_mask shape: (batch_size, n_heads, seq_len, seq_len)
        # other shape: (batch_size, n_heads, seq_len, seq_len)

        # scores shape: (batch_size, n_heads, len_q, len_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.size(3))
        if other is not None:
            scores = scores + other
        if attn_mask is not None:
            # Fills elements of self tensor with value where mask is True.
            scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        # context shape: (batch_size, n_heads, len_q, d_v)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads=1, dropout=0.5, d_v=64, d_k=64):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_v = d_v
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_Q, input_K, input_V, attn_mask=None, other=None):
        # input_Q shape: (batch_size, len_q, d_model)
        # input_K shape: (batch_size, len_k, d_model)
        # input_V sahpe: (batch_size, len_v(=len_k), d_model)
        # attn_mask shape: (batch_size, seq_len, seq_len)
        # other shape: (batch_size, seq_len, seq_len)

        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # Q shape: (batch_size, n_heads, len_q, d_k)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K shape: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V shape: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            # attn_mask shape: (batch_size, n_heads, seq_len, seq_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        if other is not None:
            other = other.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # context shape: (batch_size, n_heads, len_q, d_v)
        # attn shape: (batch_size, n_heads, len_q, len_k)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask, other)
        # context shape: (batch_size, len_q, n_heads * d_v)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        # output shape: (batch_size, len_q, d_model)
        output = self.fc(context)
        output = self.dropout(output)
        return output, attn


class GCN(torch.nn.Module):

    def __init__(self, d_model, n_layers=3):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [GCNConv(d_model, d_model) for _ in range(n_layers)])

    def forward(self, x, edge_index):
        for conv in self.conv_list:
            x = conv(x, edge_index)
            x = F.dropout(x, training=self.training)
        return x


class EmbeddingLayer(nn.Module):

    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        self.config = config

        # define embedding layer
        self.userEmbLayer = nn.Embedding(config.max_user_num, config.hidden_size, 0)
        self.locEmbLayer = nn.Embedding(config.max_loc_num, config.hidden_size, 0)
        self.geoEmbLayer = nn.Embedding(config.max_geo_num, config.hidden_size, 0)

        # init embedding layer
        nn.init.normal_(self.userEmbLayer.weight, std=0.1)
        nn.init.normal_(self.locEmbLayer.weight, std=0.1)
        nn.init.normal_(self.geoEmbLayer.weight, std=0.1)

    def forward(self, user, traj, geo, long_traj, traj_graph, geo_graph):
        #! Embedding user, traj, geohash
        # emb shape: (batch_size, max_sequence_length, hidden_size)
        user_emb = self.userEmbLayer(user)
        traj_emb = self.locEmbLayer(traj)
        geo_emb = self.geoEmbLayer(geo)

        long_traj_emb = self.locEmbLayer(long_traj)

        traj_graph.x = self.locEmbLayer(traj_graph.x)
        geo_graph.x = self.geoEmbLayer(geo_graph.x)

        return user_emb, traj_emb, geo_emb, long_traj_emb, traj_graph, geo_graph


class LocalCenterEncoder(nn.Module):

    def __init__(self, d_model, n_heads=4, dropout=0.5):
        super(LocalCenterEncoder, self).__init__()
        self.traj_conv = GCN(d_model, 3)
        self.geo_conv = GCN(d_model, 3)
        self.attn = nn.MultiheadAttention(d_model * 2, n_heads, dropout, batch_first=True)
        self.linear = nn.Linear(d_model * 2, d_model)
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()

    def forward(self, center_traj, traj_graph, geo_graph):
        # center_traj shape: (batch_size, long_sequence_length)

        # traj_conv_out/geo_conv_out shape: (batch_node_num, d_model)
        traj_conv_out = self.traj_conv(traj_graph.x, traj_graph.edge_index)
        geo_conv_out = self.geo_conv(geo_graph.x, geo_graph.edge_index)
        traj_graph.x = traj_conv_out
        geo_graph.x = geo_conv_out

        # center_traj_emb shape: (batch_size, long_sequence_length, d_model)
        center_traj = center_traj + traj_graph.ptr[:-1].unsqueeze(1)
        center_traj_emb = traj_conv_out[center_traj]

        sub_traj_graph = traj_graph.subgraph(traj_graph.freq >= traj_graph.thr)
        sub_geo_graph = geo_graph.subgraph(geo_graph.freq >= geo_graph.thr)
        # traj_personal/geo_personal shape: (batch_size, d_model)
        traj_personal = global_mean_pool(sub_traj_graph.x, sub_traj_graph.batch)
        geo_personal = global_mean_pool(sub_geo_graph.x, sub_geo_graph.batch)

        # traj_personal/geo_personal shape: (batch_size, 1, d_model)
        traj_personal = traj_personal.unsqueeze(1)
        geo_personal = geo_personal.unsqueeze(1)
        # personal shape: (batch_size, 1, d_model * 2)
        personal = torch.concat([traj_personal, geo_personal], dim=-1)

        # personal shape: (batch_size, 1, d_model)
        user_perfence, _ = self.attn(personal, personal, personal)
        user_perfence = self.linear(user_perfence)
        center_traj_emb = self.dropout1(center_traj_emb)
        user_perfence = self.dropout2(user_perfence)
        return center_traj_emb, user_perfence


class ShortTermEncoder(nn.Module):

    def __init__(self, d_model) -> None:
        super(ShortTermEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=d_model * 4,
                            hidden_size=d_model * 2,
                            batch_first=True)
        self.attn = MultiHeadAttention(d_model * 2, n_heads=4, dropout=0.5)
        self.w = nn.Parameter(torch.ones(2))
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()

    def forward(self, user_emb, traj_emb, geo_emb, center_traj_emb, long_traj_emb,
                user_perfence, dt):
        # traj_emb/geo_emb shape: (batch_size, max_sequence_length, d_model)
        # center_traj_emb shape: (batch_size, long_sequence_length, d_model)
        # long_traj_emb shape: (batch_size, long_sequence_length, d_model)
        # user_perfence shape: (batch_size, 1, d_model * 2)
        # dt shape: (batch_size, max_sequence_length, max_sequence_length)

        # user_perfence shape: (batch_size, max_sequence_length, d_model)
        user_perfence = user_perfence.repeat(1, traj_emb.size(1), 1)
        user_perfence = torch.concat([user_emb, user_perfence], dim=-1)

        # input shape: (batch_size, max_sequence_length, d_model * 4)
        input = torch.concat([traj_emb, geo_emb, user_perfence], dim=-1)

        # lstm_output shape: (batch_size, max_sequence_length, hidden_size)
        lstm_output, (hidden_state, cell_state) = self.lstm(input)
        lstm_output = self.dropout1(lstm_output)

        # center_input shape: (batch_size, long_sequence_length, d_model)
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        center_input = F.relu(w1 * long_traj_emb + w2 * center_traj_emb)

        center_input = torch.concat([center_input, \
            user_emb.repeat(1, long_traj_emb.size(1) // traj_emb.size(1), 1)], dim=-1)
        # center_out shape: (batch_size, max_sequence_length, d_model)
        center_out, _ = self.attn(lstm_output,
                                  center_input,
                                  center_input,
                                  other=(1 / (1 + dt)))

        # out shape: (batch_size, max_sequence_length, d_model)
        out = lstm_output * torch.exp(-center_out)
        out = self.dropout2(out)
        return out


class PoiModel(nn.Module):

    def __init__(self, config):
        super(PoiModel, self).__init__()
        self.config = config
        self.EmbeddingLayer = EmbeddingLayer(config)
        self.LocalCenterEncoder = LocalCenterEncoder(config.hidden_size)
        self.ShortTermEncoder = ShortTermEncoder(config.hidden_size)
        self.fc_traj = nn.Linear(config.hidden_size * 2, config.max_loc_num)
        self.fc_geo = nn.Linear(config.hidden_size * 2, config.max_geo_num)

    def forward(self, user, traj, geo, center_traj, long_traj, dt, traj_graph, geo_graph):
        # user/traj/geo shape: (batch_size, max_sequence_length)
        # center_traj/long_traj shape: (batch_size, long_sequence_length)
        # dt shape: (batch_size, max_sequence_length, max_sequence_length)

        # user_emb/traj_emb/geo_emb shape: (batch_size, max_sequence_length, hidden_size)
        user_emb, traj_emb, geo_emb, long_traj_emb, traj_graph, geo_graph = self.EmbeddingLayer(
            user, traj, geo, long_traj, traj_graph, geo_graph)

        # user_perfence shape: (batch_size, 1, hidden_size)
        # center_traj_emb shape: (batch_size, long_sequence_length, hidden_size)
        center_traj_emb, user_perfence = self.LocalCenterEncoder(
            center_traj, traj_graph, geo_graph)

        # short_enc_out shape: (batch_size, max_sequence_length, hidden_size)
        short_enc_out = self.ShortTermEncoder(user_emb, traj_emb, geo_emb,
                                              center_traj_emb, long_traj_emb,
                                              user_perfence, dt)

        # pred_traj shape: (batch_size, max_sequence_length, max_loc_num)
        # pred_geo shape: (batch_size, max_sequence_length, max_geo_num)
        pred_traj = self.fc_traj(short_enc_out)
        pred_geo = self.fc_geo(short_enc_out)

        return pred_traj, pred_geo
