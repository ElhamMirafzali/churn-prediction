import pandas as pd
from sklearn import preprocessing
from datetime import date


def preprocess_transactions(data_path):
    data = pd.read_csv(data_path)

    # filling missing values with 0
    data = data.fillna(0)

    # normalization
    data[['payment_plan_days',
          'plan_list_price',
          'actual_amount_paid']] = preprocessing.normalize(data[['payment_plan_days',
                                                                 'plan_list_price',
                                                                 'actual_amount_paid']])
    # one hot encoding
    data = pd.get_dummies(data, columns=["payment_method_id"], prefix=["payment_method"])

    return data


def avg_time_between_trans(dataset):
    for i, row in dataset.iterrows():
        if i == 0:
            dataset.loc[i, 'avg_time_between_trans'] = 0
        else:
            date1 = str(dataset.loc[i - 1, 'transaction_date'])
            date2 = str(dataset.loc[i, 'transaction_date'])
            date1 = date(year=int(date1[0:4]), month=int(date1[4:6]), day=int(date1[6:8]))
            date2 = date(year=int(date2[0:4]), month=int(date2[4:6]), day=int(date2[6:8]))
            dataset.loc[i, 'avg_time_between_trans'] = (((date2 - date1).days + (
                    dataset.loc[i - 1, 'avg_time_between_trans'] * (i - 1))) / i)
    # normalize new column
    dataset[['avg_time_between_trans']] = preprocessing.normalize(dataset[['avg_time_between_trans']])
    return dataset
