import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoBranchesLSTMModel(nn.Module):

    def __init__(self, input_size_x1, input_size_x2, output_size, hidden_dim_lstm, num_layers, fc1_units, fc2_units,
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
        self.fc1 = nn.Linear(64, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # print("x shape = ", x.shape)
        out_lstm1, hidden1 = self.lstm1(x1, (self.hidden_state1, self.cell_state1))
        self.hidden_state1 = hidden1[0].detach()
        self.cell_state1 = hidden1[1].detach()
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # out = self.dropout(lstm_out)

        out_lstm2, hidden2 = self.lstm2(x2, (self.hidden_state2, self.cell_state2))
        self.hidden_state2 = hidden2[0].detach()
        self.cell_state2 = hidden2[1].detach()

        # print("lstm1_out = ", out_lstm1.shape)
        # print("lstm2_out = ", out_lstm2.shape)

        concat_hidden = torch.cat(tensors=[out_lstm1[:, -1, :], out_lstm2[:, -1, :]], dim=1)

        # h2_t_1 = torch.unsqueeze(self.hidden_state1[1, 0, -1], 0)
        # h2_t_2 = torch.unsqueeze(self.hidden_state2[1, 0, -1], 0)
        # concat_hidden = torch.cat(tensors=[h2_t_1, h2_t_2], dim=0)

        # print("self.concat_hidden.shape = ", concat_hidden.shape)

        out_fc1 = F.relu(self.fc1(concat_hidden))
        out_fc2 = F.relu(self.fc2(out_fc1))
        out_fc3 = F.relu(self.fc3(out_fc2))
        out = self.sigmoid(out_fc3)

        # apply sigmoid function to fc_out to get the probability
        # out = self.sigmoid(fc_out)
        # lstm_out[:, -1, :]
        return out

    def init_hidden(self, batch_size):
        hidden_state = torch.randn(self.num_layers, batch_size, self.hidden_dim).double()
        cell_state = torch.randn(self.num_layers, batch_size, self.hidden_dim).double()
        return hidden_state, cell_state
