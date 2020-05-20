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
          'actual_amount_paid',
          'avg_time_between_trans']] = preprocessing.normalize(data[['payment_plan_days',
                                                                     'plan_list_price',
                                                                     'actual_amount_paid',
                                                                     'avg_time_between_trans']], axis=0)

    # one hot encoding
    data = pd.get_dummies(data, columns=["payment_method_id"], prefix=["payment_method"])

    train_trans_normalized = data[0:len(train_trans)]
    test_trans_normalized = data[len(train_trans):]

    train_trans_normalized.to_csv(
        'new_data/selected2/new_normalization/train_transactions_with_extracted_features_preprocessed.csv',
        index=False)
    test_trans_normalized.to_csv(
        'new_data/selected2/new_normalization/test_transactions_with_extracted_features_preprocessed.csv',
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
                                                                         'days_since_last_log_of_100']], axis=0)

    train_logs_normalized = data[0:len(train_logs)]
    test_logs_normalized = data[len(train_logs):]

    train_logs_normalized.to_csv(
        'new_data/selected2/new_normalization/train_logs_with_extracted_features_preprocessed.csv', index=False)
    test_logs_normalized.to_csv(
        'new_data/selected2/new_normalization/test_logs_with_extracted_features_preprocessed.csv', index=False)
    return train_logs_normalized, test_logs_normalized


def preprocess_members(data_path):
    data = pd.read_csv(data_path)

    data = data[data.bd < 200]
    data = data[data.bd >= 0]

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

    data.to_csv('data/members_age_filtered_preprocessed.csv', index=False)
    return data

# preprocess_logs(train_path='new_data/selected2/train_logs_with_extracted_features.csv',
#                 test_path='new_data/selected2/test_logs_with_extracted_features.csv')

# preprocess_members(data_path='data/members.csv')

# preprocess_transactions(train_path='new_data/selected2/train_transactions_with_extracted_features.csv',
#                         test_path='new_data/selected2/test_transactions_with_extracted_features.csv')


