import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings(action='ignore')

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, multi=True):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.multi = multi

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        if self.multi:
            last_output = rnn_output[:, -1, :]
        else:
            last_output = rnn_output[:, -1]

        final_output = self.fc(last_output)
        return final_output

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_size, multi=True):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.multi = multi

    def forward(self, x):

        lstm_output, _ = self.lstm(x)
        if self.multi:
            last_output = lstm_output[:, -1, :]
        else:
            last_output = lstm_output[:, -1]

        output = self.fc(last_output)

        return output

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_size, multi):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.multi = multi

    def forward(self, x):
        gru_output, _ = self.gru(x)

        if self.multi:
            last_output = gru_output[:, -1, :]
        else:
            last_output = gru_output[:, -1]

        output = self.fc(last_output)

        return output

class TsTrans(nn.Module):
    def __init__(self, input_dim=11, max_len=100, output_dim=1, d_model=512, nhead=8, num_layers=1, dropout=0.1):
        super(TsTrans, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = np.array([0])
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        if (self.src_mask == None) | ( len(self.src_mask) != len(src)):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask[:src.shape[0], :src.shape[0]]

        src = self.embedding(src)
        src = self.pos_encoder(src)
        en_output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(en_output)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
