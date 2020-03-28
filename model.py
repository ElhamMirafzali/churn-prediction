import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, num_layers, batch_size, dropout):
        super(LSTMModel, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)

        self.hidden_state, self.cell_state = self.init_hidden(batch_size=batch_size)
        # self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("x shape = ", x.shape)
        lstm_out, hidden = self.lstm(x, (self.hidden_state, self.cell_state))
        self.hidden_state = hidden[0].detach()
        self.cell_state = hidden[1].detach()
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # out = self.dropout(lstm_out)

        fc_out = self.fc(lstm_out)
        # apply sigmoid function to fc_out to get the probability
        out = self.sigmoid(fc_out)

        return out[:, -1, :], hidden

    def init_hidden(self, batch_size):
        hidden_state = torch.randn(self.num_layers, batch_size, self.hidden_dim).double()
        cell_state = torch.randn(self.num_layers, batch_size, self.hidden_dim).double()
        return hidden_state, cell_state
