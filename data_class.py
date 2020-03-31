import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
from preprocess import preprocess_transactions, avg_time_between_trans


class SequentialDataset(Dataset):
    def __init__(self, transactions_path, targets_path):
        self.transactions = preprocess_transactions(transactions_path)
        self.targets = (pd.read_csv(targets_path)).sample(frac=1)
        self.users = self.targets['msno']

    def __getitem__(self, index: int):
        dataset: pd.DataFrame = self.transactions.loc[self.transactions['msno'] == self.users[index]]
        dataset = dataset.sort_values('transaction_date')
        dataset.reset_index(drop=True, inplace=True)
        target = torch.tensor((self.targets.loc[self.targets['msno'] == self.users[index]]['is_churn']).squeeze(),
                              dtype=torch.double)

        # add a column: average time between transactions
        dataset = avg_time_between_trans(dataset)

        # 'msno', 'transaction_date', 'membership_expire_date' are removed
        dataset = dataset[dataset.columns.difference(['msno', 'transaction_date', 'membership_expire_date'])]

        tensor_data = torch.tensor(data=dataset.values, dtype=torch.double)

        return tensor_data, target

    def __len__(self):
        return len(self.users)
