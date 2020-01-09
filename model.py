import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)

        # self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)

        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        out = self.sigmoid(out)

        return out, hidden

    def init_hidden(self, batch_size):
        hidden_state = torch.randn(self.num_layers, batch_size, self.hidden_dim)
        cell_state = torch.randn(self.num_layers, batch_size, self.hidden_dim)
        hidden = (hidden_state, cell_state)
        return hidden
