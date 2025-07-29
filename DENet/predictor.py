import torch
import torch.nn as nn
import math
import collections
import numpy as np
from torch.nn.parameter import Parameter

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros((max_len, d_model), requires_grad=False).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class InputPositionEmbedding(nn.Module):
    def __init__(self, vocab_size=None, embed_dim=None, dropout=0.1,
                init_weight=None, seq_len=None):
        super(InputPositionEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.position_embed = PositionalEmbedding(embed_dim, max_len=seq_len)
        self.reproject = nn.Identity()
        if init_weight is not None:
            self.embed = nn.Embedding.from_pretrained(init_weight)
            self.reproject = nn.Linear(init_weight.size(1), embed_dim)

    def forward(self, inputs):
        x = self.embed(inputs)
        x = x + self.position_embed(inputs)
        x = self.reproject(x)
        x = self.dropout(x)
        return x


class AggregateLayer(nn.Module):
    def __init__(self, d_model=None, dropout=0.1, attention_dim=None):
        super(AggregateLayer, self).__init__()
        self.attn = nn.Sequential(collections.OrderedDict([
            ('layernorm', nn.LayerNorm(d_model)),
            ('fc1', nn.Linear(d_model, attention_dim)),
            ('tanh', nn.Tanh()),
            ('fc2', nn.Linear(attention_dim, 1, bias=False)),
            ('dropout', nn.Dropout(dropout)),
            ('softmax', nn.Softmax(dim=1))
        ]))

    def forward(self, context):
        weight = self.attn(context)
        output = torch.bmm(context.transpose(1, 2), weight)
        output = output.squeeze(2)
        return output


class GlobalPredictor(nn.Module):
    def __init__(self, d_model=None, d_h=None, d_out=None, dropout=0.5):
        super(GlobalPredictor, self).__init__()
        self.predict_layer = nn.Sequential(collections.OrderedDict([
            ('batchnorm', nn.BatchNorm1d(d_model)),
            ('fc1', nn.Linear(d_model, d_h)),
            ('tanh', nn.Tanh()),
            ('dropout', nn.Dropout(dropout)),
            ('fc2', nn.Linear(d_h, d_out))
        ]))

    def forward(self, x):
        x = self.predict_layer(x)
        return x


class SequenceLSTM(nn.Module):
    def __init__(self, d_input=None, d_embed=20, d_model=128,
                vocab_size=None, seq_len=None,
                dropout=0.1, lstm_dropout=0,
                nlayers=1, bidirectional=False,
                proj_loc_config=None):
        super(SequenceLSTM, self).__init__()

        self.embed = InputPositionEmbedding(vocab_size=vocab_size,
                    seq_len=seq_len, embed_dim=d_embed)

        self.lstm = nn.LSTM(input_size=d_input,
                            hidden_size=d_model//2 if bidirectional else d_model,
                            num_layers=nlayers, dropout=lstm_dropout,
                            bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        self.proj_loc_layer = proj_loc_config['layer'](
                    proj_loc_config['d_in'], proj_loc_config['d_out']
                )

    def forward(self, x, loc_feat=None):
        x = self.embed(x)
        if loc_feat is not None:
            loc_feat = self.proj_loc_layer(loc_feat)
            x = torch.cat([x, loc_feat], dim=2)
        x = x.transpose(0, 1).contiguous()
        x, _ = self.lstm(x)
        x = x.transpose(0, 1).contiguous()
        x = self.drop(x)
        return x


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, edge_map, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        initial_value = torch.from_numpy(edge_map).float()
        self.edge = torch.nn.Parameter(initial_value.clone().detach().requires_grad_(True))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = input @ self.weight  
        output = self.edge @ support  
        if self.bias is not None:  
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Attention(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  # input.shape = (seq_len, input_dim)
        x = torch.tanh(self.fc1(input))  # x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)  # x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  # attention.shape = (1, attention_hops, seq_len)
        return attention


class DENetPredictor(nn.Module):
    def __init__(self, d_embed=20, d_model=128, out_dim=512, d_h=32, d_out=1, d_attn=3,
                vocab_size=None, seq_len=None, use_gcn=True,
                dropout=0.1, lstm_dropout=0, nlayers=1, bidirectional=False,
                use_loc_feat=True, use_glob_feat=True,
                proj_loc_config=None, proj_glob_config=None):
        super(DENetPredictor, self).__init__()
        self.seq_lstm = SequenceLSTM(
            d_input=d_embed + (proj_loc_config['d_out'] if use_loc_feat else 0),
            d_embed=d_embed, d_model=d_model,
            vocab_size=vocab_size, seq_len=seq_len,
            dropout=dropout, lstm_dropout=lstm_dropout,
            nlayers=nlayers, bidirectional=bidirectional,
            proj_loc_config=proj_loc_config)
        self.proj_glob_layer = proj_glob_config['layer'](
            proj_glob_config['d_in'], proj_glob_config['d_out'])
        dim = d_model + (proj_glob_config['d_out'] if use_glob_feat else 0)

        self.weight = Parameter(torch.FloatTensor(dim, out_dim))
        self.bias = Parameter(torch.FloatTensor(out_dim))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        self.ln = nn.LayerNorm(out_dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.compress = nn.Linear(dim, out_dim)
        self.aggragator = AggregateLayer(d_model=out_dim, attention_dim=int(out_dim/d_attn))
        self.predictor = GlobalPredictor(
            d_model=out_dim, d_h=d_h, d_out=d_out)
        self.use_gcn = use_gcn

    def forward(self, x, edge, cluster=None, glob_feat=None, loc_feat=None):
            x = self.seq_lstm(x, loc_feat=loc_feat)
            x = x.float()
            if glob_feat is not None:
                glob_feat = self.proj_glob_layer(glob_feat)
                x = torch.cat([x, glob_feat], dim=2)
            if self.use_gcn:
                support = x @ self.weight  
                batch_size = x.size(0)
                edge_index=[]
                for i in range(batch_size):
                    #edge_index.append(cluster[i])
                    edge_index.append(0)
                edges = torch.stack([edge[i] for i in edge_index])
                x = edges @ support
                x = x + self.bias
                x = self.relu(self.ln(x))
            else:
                x = self.compress(x)
            x = self.aggragator(x)
            output = self.predictor(x)
            return output


if __name__ == "__main__":
    c_map = np.random.rand(500, 500)
    model = DENetPredictor(
        d_embed = 20, d_model = 128, hid_dim = 512, out_dim = 256, d_h = 128, d_out = 1,
        vocab_size=21, seq_len=500, edge_map = c_map,
        use_loc_feat = True, use_glob_feat = True,
        proj_glob_config = {'layer':nn.Linear, 'd_in':1536, 'd_out':512},
        proj_loc_config = {'layer':nn.Linear, 'd_in':500, 'd_out':128},
        )
    x = torch.randint(0, 21, (128, 500))
    a = x
    glob_feat = torch.rand((128, 500, 1536))
    loc_feat = torch.rand((128, 500, 500))
    adj = torch.rand(500, 500)
    y = model(x, glob_feat=glob_feat, loc_feat=loc_feat)
