import torch
from torch.utils.data.dataset import Dataset
import pandas as pd


class SequentialDataset(Dataset):
    def __init__(self, transactions_path, targets_path):
        self.transactions = pd.read_csv(transactions_path)
        self.targets = (pd.read_csv(targets_path)).sample(frac=1)
        self.users = self.targets['msno']

    def __getitem__(self, index: int):
        dataset = self.transactions.loc[self.transactions['msno'] == self.users[index]]
        dataset = dataset.sort_values('transaction_date')
        target = torch.tensor((self.targets.loc[self.targets['msno'] == self.users[index]]['is_churn']).squeeze(),
                              dtype=torch.double)
        tensor_data = torch.tensor(data=dataset[[
            'payment_method_id',
            'payment_plan_days',
            'plan_list_price',
            'actual_amount_paid',
            'is_auto_renew',
            'transaction_date',
            'membership_expire_date',
            'is_cancel']].values, dtype=torch.double)

        return tensor_data, target

    def __len__(self):
        return len(self.users)
