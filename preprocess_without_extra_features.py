import pandas as pd
from sklearn import preprocessing


def preprocess_transactions(train_path, test_path):
    train_trans = pd.read_csv(train_path)
    test_trans = pd.read_csv(test_path)
    data = pd.concat([train_trans, test_trans], ignore_index=True)

    # filling missing values with 0
    data = data.fillna(0)

    # normalization
    data[['payment_plan_days',
          'plan_list_price',
          'actual_amount_paid']] = preprocessing.normalize(data[['payment_plan_days',
                                                                 'plan_list_price',
                                                                 'actual_amount_paid']], axis=0)

    # one hot encoding
    data = pd.get_dummies(data, columns=["payment_method_id"], prefix=["payment_method"])

    train_trans_normalized = data[0:len(train_trans)]
    test_trans_normalized = data[len(train_trans):]

    train_trans_normalized.to_csv(
        'data/train_selected/without_extra_features/train_transactions_preprocessed.csv',
        index=False)
    test_trans_normalized.to_csv(
        'data/test_selected/without_extra_features/test_transactions_preprocessed.csv',
        index=False)

    return train_trans_normalized, test_trans_normalized


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
          'total_secs']] = preprocessing.normalize(data[['num_25',
                                                         'num_50',
                                                         'num_75',
                                                         'num_985',
                                                         'num_100',
                                                         'num_unq',
                                                         'total_secs']], axis=0)

    train_logs_normalized = data[0:len(train_logs)]
    test_logs_normalized = data[len(train_logs):]

    train_logs_normalized.to_csv(
        'data/train_selected/without_extra_features/train_logs_preprocessed.csv', index=False)
    test_logs_normalized.to_csv(
        'data/test_selected/without_extra_features/test_logs_preprocessed.csv', index=False)
    return train_logs_normalized, test_logs_normalized


def preprocess_members(train_path, test_path):
    train_members = pd.read_csv(train_path)
    test_members = pd.read_csv(test_path)
    data = pd.concat([train_members, test_members], ignore_index=True)

    # Label Encoding (categorical to numeric)
    data['gender'] = data['gender'].astype('category')
    data['gender'] = data['gender'].cat.codes
    # rows which did not specified the gender becomes (gender = -1)
    # female -> 0
    # male -> 1

    # filling missing values with 0
    data = data.fillna(0)

    # normalization
    data['bd'] = data['bd'].astype(int)
    data[['bd']] = preprocessing.normalize(data[['bd']], axis=0)

    # one hot encoding
    data = pd.get_dummies(data, columns=["city"], prefix=["city"])
    data = pd.get_dummies(data, columns=["registered_via"], prefix=["registered_via"])

    train_members_normalized = data[0:len(train_members)]
    test_members_normalized = data[len(test_members):]

    train_members_normalized.to_csv(
        'data/train_selected/without_extra_features/train_members_preprocessed.csv', index=False)
    test_members_normalized.to_csv(
        'data/test_selected/without_extra_features/test_members_preprocessed.csv', index=False)
    return train_members_normalized, test_members_normalized


# preprocess_members(train_path='data/train_selected/train_members.csv',
#                    test_path='data/test_selected/test_members.csv')

# preprocess_transactions(train_path='data/train_selected/train_transactions(membership_expires_in_feb).csv',
#                         test_path='data/test_selected/test_transactions(membership_expires_in_march).csv')

# preprocess_logs(train_path='data/train_selected/train_logs_dec2016_to_feb2017.csv',
#                 test_path='data/test_selected/test_logs_jan2017_to_march2017.csv')
