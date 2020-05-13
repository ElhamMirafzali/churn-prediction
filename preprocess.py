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


def preprocess_logs(train_path, test_path):
    train_logs = pd.read_csv(train_path)
    test_logs = pd.read_csv(test_path)
    data = pd.concat([train_logs, test_logs], ignore_index=True)

    # filling missing values with 0
    data = data.fillna(0)

    # normalization
    data[['num_25',
          'num_50',
          'num_75',
          'num_985',
          'num_100',
          'num_unq',
          'total_secs',
          'days_since_last_log_of_25',
          'days_since_last_log_of_50',
          'days_since_last_log_of_75',
          'days_since_last_log_of_985',
          'days_since_last_log_of_100']] = preprocessing.normalize(data[['num_25',
                                                                         'num_50',
                                                                         'num_75',
                                                                         'num_985',
                                                                         'num_100',
                                                                         'num_unq',
                                                                         'total_secs',
                                                                         'days_since_last_log_of_25',
                                                                         'days_since_last_log_of_50',
                                                                         'days_since_last_log_of_75',
                                                                         'days_since_last_log_of_985',
                                                                         'days_since_last_log_of_100']])

    train_logs_normalized = data[0:len(train_logs)]
    test_logs_normalized = data[len(train_logs):]

    train_logs_normalized.to_csv('new_data/selected2/train_logs_with_extracted_features_preprocessed.csv', index=False)
    test_logs_normalized.to_csv('new_data/selected2/test_logs_with_extracted_features_preprocessed.csv', index=False)
    return train_logs_normalized, test_logs_normalized


def preprocess_members(data_path):
    data = pd.read_csv(data_path)

    # Label Encoding (categorical to numeric)
    data['gender'] = data['gender'].astype('category')
    data['gender'] = data['gender'].cat.codes
    # rows which did not specified the gender becomes (gender = -1)
    # female -> 0
    # male -> 1

    # filling missing values with 0
    data = data.fillna(0)

    # normalization
    data[['bd']] = preprocessing.normalize(data[['bd']])

    # one hot encoding
    data = pd.get_dummies(data, columns=["city"], prefix=["city"])
    data = pd.get_dummies(data, columns=["registered_via"], prefix=["registered_via"])

    data.to_csv('data/members_preprocessed.csv', index=False)
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

# preprocess_logs(train_path='new_data/selected2/train_logs_with_extracted_features.csv',
#                 test_path='new_data/selected2/test_logs_with_extracted_features.csv')

# preprocess_members(data_path='data/members.csv')
