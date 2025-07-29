import torch
import torch.nn as nn
import collections
import numpy as np

SEED = 2333
np.random.seed(SEED)
torch.manual_seed(SEED)

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
    def __init__(self, out_dim=1536, d_h=128, d_out=1, d_len=56, d_attn=1):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(d_len, 1)
        self.predictor = GlobalPredictor(
            d_model=out_dim, d_h=d_h, d_out=d_out)

    def forward(self, x):
            x = x.float()
            x = x.transpose(1, 2) 
            x = self.fc(x)
            x = x.squeeze(-1) 
            output = self.predictor(x)
            return output


if __name__ == "__main__":
    model = Predictor(
        out_dim = 1536, d_h = 128, d_out = 1
        )
    x = torch.rand((128, 500, 1536))
    y = model(x)
    print(y)
