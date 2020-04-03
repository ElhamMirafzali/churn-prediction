import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
from preprocess import preprocess_transactions, avg_time_between_trans, preprocess_logs


class SequentialDataset(Dataset):
    def __init__(self, transactions_path, logs_path, targets_path):
        self.transactions = preprocess_transactions(transactions_path)
        self.targets = (pd.read_csv(targets_path)).sample(frac=1)
        self.users = self.targets['msno']
        self.logs = preprocess_logs(logs_path)

    def __getitem__(self, index: int):
        data_trans: pd.DataFrame = self.transactions.loc[self.transactions['msno'] == self.users[index]]
        data_trans = data_trans.sort_values('transaction_date')
        data_trans.reset_index(drop=True, inplace=True)
        target = torch.tensor((self.targets.loc[self.targets['msno'] == self.users[index]]['is_churn']).squeeze(),
                              dtype=torch.double)

        # add a column: average time between transactions
        data_trans = avg_time_between_trans(data_trans)

        # 'msno', 'transaction_date', 'membership_expire_date' are removed
        data_trans = data_trans[data_trans.columns.difference(['msno', 'transaction_date', 'membership_expire_date'])]

        # transactions
        tensor_trans = torch.tensor(data=data_trans.values, dtype=torch.double)

        # logs
        data_logs: pd.DataFrame = self.logs.loc[self.logs['msno'] == self.users[index]]
        data_logs = data_logs.sort_values('date')
        data_logs.reset_index(drop=True, inplace=True)
        data_logs = data_logs[data_logs.columns.difference(['msno', 'date'])]
        tensor_logs = torch.tensor(data=data_logs.values, dtype=torch.double)

        return tensor_trans, tensor_logs, target

    def __len__(self):
        return len(self.users)
