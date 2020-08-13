import torch
from torch.utils.data.dataset import Dataset
import pandas as pd


class SequentialDataset(Dataset):
    def __init__(self, transactions, logs, members, targets):
        self.targets = targets
        self.transactions = transactions
        # self.targets = (pd.read_csv(targets_path)).sample(frac=1)
        self.users = self.targets['msno']
        self.logs = logs
        self.members = members

    def __getitem__(self, index: int):
        # print("user = ", self.users[index])
        # target
        target = torch.tensor([(self.targets.loc[self.targets['msno'] == self.users[index]]['is_churn']).squeeze()],
                              dtype=torch.double)

        # transactions
        data_trans: pd.DataFrame = self.transactions.loc[self.transactions['msno'] == self.users[index]]
        data_trans = data_trans.sort_values('transaction_date')
        data_trans.reset_index(drop=True, inplace=True)
        # 'msno', 'transaction_date', 'membership_expire_date' are removed
        data_trans = data_trans[data_trans.columns.difference(['msno', 'transaction_date', 'membership_expire_date'])]
        data_trans = data_trans[data_trans.columns.difference(
            ['days_since_registration', 'non_subscribed_rate'])]
        tensor_trans = torch.tensor(data=data_trans.values, dtype=torch.double)

        # logs
        data_logs: pd.DataFrame = self.logs.loc[self.logs['msno'] == self.users[index]]
        data_logs = data_logs.sort_values('date')
        data_logs.reset_index(drop=True, inplace=True)
        data_logs = data_logs[data_logs.columns.difference(['msno', 'date'])]
        data_logs = data_logs[data_logs.columns.difference(['msno', 'date', 'days_since_registration'])]
        tensor_logs = torch.tensor(data=data_logs.values, dtype=torch.double)

        # members
        data_members: pd.DataFrame = self.members.loc[self.members['msno'] == self.users[index]]
        data_members = data_members[data_members.columns.difference(['msno', 'registration_init_time'])]
        tensor_members = torch.tensor(data=data_members.values, dtype=torch.double)

        return tensor_trans, tensor_logs, tensor_members, target

    def __len__(self):
        return len(self.users)
