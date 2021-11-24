import torch
#import torch.nn as nn
from torch import nn, optim

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hs):
   
        out, hs = self.lstm(x, hs)           # out.shape = (batch_size, seq_len, hidden_size)
        out = out.view(-1, self.hidden_size) # out.shape = (seq_len, hidden_size)     
        out = self.fc(out)
        
        return out, hs