import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        input_size = int(cfg.IN_HORIZON * cfg.SAMPLING_RATE + cfg.OUT_HORIZON * cfg.SAMPLING_RATE)
        hidden_size = cfg.LSTM_NODES

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
            
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.batchnorm(x)
        x = self.linear(x)
        x = F.sigmoid(x).flatten()
        return x