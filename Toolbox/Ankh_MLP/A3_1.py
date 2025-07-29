import torch
import torch.nn as nn
import collections
import numpy as np

SEED = 2333
np.random.seed(SEED)
torch.manual_seed(SEED)

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
        '''
        Parameters
        ----------
        context: token embedding from encoder (Transformer/LSTM)
                (batch_size, seq_len, embed_dim)
        '''

        weight = self.attn(context)
        # (batch_size, seq_len, embed_dim).T * (batch_size, seq_len, 1) *  ->
        # (batch_size, embed_dim, 1)
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



class Predictor(nn.Module):
    def __init__(self, out_dim=1536, d_h=128, d_out=1, d_attn=3):
        super(Predictor, self).__init__()
        self.aggragator = AggregateLayer(d_model=out_dim, attention_dim=int(out_dim/d_attn))
        self.predictor = GlobalPredictor(
            d_model=out_dim, d_h=d_h, d_out=d_out)

    def forward(self, x):
            x = x.float()
            x = self.aggragator(x)
            output = self.predictor(x)
            return output


if __name__ == "__main__":
    model = Predictor(
        out_dim = 1536, d_h = 128, d_out = 1
        )
    x = torch.rand((128, 500, 1536))
    y = model(x)
    print(y)
