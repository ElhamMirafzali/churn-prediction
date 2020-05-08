import torch
from torch.utils.data.dataset import Dataset
import pandas as pd


class LogsSequentialDataset(Dataset):
    def __init__(self, logs, targets_path):
        self.targets = (pd.read_csv(targets_path)).sample(frac=1)
        # self.users = self.targets['msno']
        self.users = logs.msno.unique()
        self.logs = logs

    def __getitem__(self, index: int):
        target = torch.tensor([(self.targets.loc[self.targets['msno'] == self.users[index]]['is_churn']).squeeze()],
                              dtype=torch.double)

        # logs
        data_logs: pd.DataFrame = self.logs.loc[self.logs['msno'] == self.users[index]]
        data_logs = data_logs.sort_values('date')
        data_logs.reset_index(drop=True, inplace=True)
        data_logs = data_logs[data_logs.columns.difference(['msno', 'date'])]
        tensor_logs = torch.tensor(data=data_logs.values, dtype=torch.double)

        return tensor_logs, target

    def __len__(self):
        return len(self.users)
