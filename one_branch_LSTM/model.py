import torch
import torch.nn as nn


class OneBranchLSTMModel(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, num_layers, fc1_units, fc2_units, batch_size, dropout):
        super(OneBranchLSTMModel, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)

        self.hidden_state, self.cell_state = self.init_hidden(batch_size=batch_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        self.hidden_state = hidden[0].detach()
        self.cell_state = hidden[1].detach()
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # dropout_out = self.dropout(lstm_out)

        fc1_out = self.tanh(self.fc1(lstm_out))
        fc2_out = self.tanh(self.fc2(fc1_out))
        fc3_out = self.tanh(self.fc3(fc2_out))
        # fc1_out = self.fc1(lstm_out)
        # fc2_out = self.fc2(fc1_out)
        # fc3_out = self.fc3(lstm_out)
        out = self.sigmoid(fc3_out)
        return out[:, -1, :], hidden

    def init_hidden(self, batch_size):
        hidden_state = torch.randn(self.num_layers, batch_size, self.hidden_dim).double()
        cell_state = torch.randn(self.num_layers, batch_size, self.hidden_dim).double()
        # h0 = nn.init.xavier_normal_(hidden_state)
        # c0 = nn.init.xavier_normal_(cell_state)
        return hidden_state, cell_state
