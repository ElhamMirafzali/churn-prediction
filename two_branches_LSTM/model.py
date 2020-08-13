import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoBranchesLSTMModel(nn.Module):

    def __init__(self, input_size_x1, input_size_x2, input_size_x3, output_size, hidden_dim_lstm, num_layers,
                 fc0_units, fc1_units, fc2_units,
                 batch_size, dropout):
        super(TwoBranchesLSTMModel, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim_lstm
        self.num_layers = num_layers
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units

        self.lstm1 = nn.LSTM(input_size=input_size_x1,
                             hidden_size=hidden_dim_lstm,
                             num_layers=num_layers,
                             dropout=dropout,
                             batch_first=True)

        self.lstm2 = nn.LSTM(input_size=input_size_x2,
                             hidden_size=hidden_dim_lstm,
                             num_layers=num_layers,
                             dropout=dropout,
                             batch_first=True)

        self.hidden_state1, self.cell_state1 = self.init_hidden(batch_size=batch_size)
        self.hidden_state2, self.cell_state2 = self.init_hidden(batch_size=batch_size)
        # self.dropout = nn.Dropout(dropout)
        self.fc0 = nn.Linear(input_size_x3, fc0_units)
        self.fc1 = nn.Linear(32*3, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        # print("x shape = ", x.shape)
        lstm1_out, hidden1 = self.lstm1(x1)
        self.hidden_state1 = hidden1[0].detach()
        self.cell_state1 = hidden1[1].detach()
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # out = self.dropout(lstm_out)

        lstm2_out, hidden2 = self.lstm2(x2)
        self.hidden_state2 = hidden2[0].detach()
        self.cell_state2 = hidden2[1].detach()

        fc0_out = self.fc0(x3)

        # print("lstm1_out.shape = ", lstm1_out.shape)
        # print("lstm2_out.shape = ", lstm2_out.shape)
        # print("members = ", x3)
        # print("fc0_out.shape = ", fc0_out.shape)

        concat_hidden = torch.cat(tensors=[lstm1_out[:, -1, :], lstm2_out[:, -1, :], fc0_out[:, -1, :]], dim=1)
        # print("concat_hidden.shape = ", concat_hidden.shape)

        # h2_t_1 = torch.unsqueeze(self.hidden_state1[1, 0, -1], 0)
        # h2_t_2 = torch.unsqueeze(self.hidden_state2[1, 0, -1], 0)
        # concat_hidden = torch.cat(tensors=[h2_t_1, h2_t_2], dim=0)

        fc1_out = self.tanh(self.fc1(concat_hidden))
        fc2_out = self.tanh(self.fc2(fc1_out))
        fc3_out = self.tanh(self.fc3(fc2_out))
        out = self.sigmoid(fc3_out)

        # apply sigmoid function to fc_out to get the probability
        # out = self.sigmoid(fc_out)
        # lstm_out[:, -1, :]
        return out

    def init_hidden(self, batch_size):
        hidden_state = torch.randn(self.num_layers, batch_size, self.hidden_dim).double()
        cell_state = torch.randn(self.num_layers, batch_size, self.hidden_dim).double()
        return hidden_state, cell_state
