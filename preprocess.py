import pandas as pd
from sklearn import preprocessing


def preprocess_transactions(train_path, test_path):
    train_trans = pd.read_csv(train_path)
    test_trans = pd.read_csv(test_path)
    data = pd.concat([train_trans, test_trans], ignore_index=True)

    # days_since_last_cancellation NAN -> 1
    data[['days_since_last_cancellation']] = data[['days_since_last_cancellation']].fillna(1)

    # filling missing values with 0
    data = data.fillna(0)

    # normalization
    data[['payment_plan_days',
          'plan_list_price',
          'actual_amount_paid',
          'avg_time_between_trans',
          'days_since_registration']] = preprocessing.normalize(data[['payment_plan_days',
                                                                      'plan_list_price',
                                                                      'actual_amount_paid',
                                                                      'avg_time_between_trans',
                                                                      'days_since_registration']], axis=0)

    # one hot encoding
    data = pd.get_dummies(data, columns=["payment_method_id"], prefix=["payment_method"])

    train_trans_normalized = data[0:len(train_trans)]
    test_trans_normalized = data[len(train_trans):]

    train_trans_normalized.to_csv(
        'new_data/selected2/extra_features/train_transactions_extracted_features_preprocessed.csv',
        index=False)
    test_trans_normalized.to_csv(
        'new_data/selected2/extra_features/test_transactions_extracted_features_preprocessed.csv',
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
          'days_since_last_log_of_100',
          'days_since_registration']] = preprocessing.normalize(data[['num_25',
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
                                                                      'days_since_last_log_of_100',
                                                                      'days_since_registration']], axis=0)

    train_logs_normalized = data[0:len(train_logs)]
    test_logs_normalized = data[len(train_logs):]

    train_logs_normalized.to_csv(
        'new_data/selected2/extra_features/train_logs_extracted_features_preprocessed.csv', index=False)
    test_logs_normalized.to_csv(
        'new_data/selected2/extra_features/test_logs_extracted_features_preprocessed.csv', index=False)
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
    data['bd'] = data['bd'].astype(int)
    data[['bd']] = preprocessing.normalize(data[['bd']], axis=0)

    # one hot encoding
    data = pd.get_dummies(data, columns=["city"], prefix=["city"])
    data = pd.get_dummies(data, columns=["registered_via"], prefix=["registered_via"])

    data.to_csv('data/members_age_filtered_preprocessed.csv', index=False)
    return data


def preprocess_static_data(train_path, test_path):
    train_logs = pd.read_csv(train_path)
    test_logs = pd.read_csv(test_path)
    data = pd.concat([train_logs, test_logs], ignore_index=True)

    # filling missing values with 0
    data = data.fillna(0)

    # normalization
    data[['num_25_mean',
          'num_50_mean',
          'num_75_mean',
          'num_985_mean',
          'num_100_mean',
          'num_unq_mean',
          'total_secs_mean',
          'num_25_sum',
          'num_50_sum',
          'num_75_sum',
          'num_985_sum',
          'num_100_sum',
          'num_unq_sum',
          'total_secs_sum',
          'num_25_max',
          'num_50_max',
          'num_75_max',
          'num_985_max',
          'num_100_max',
          'num_unq_max',
          'total_secs_max',
          'subscription_ratio']] = preprocessing.normalize(data[['num_25_mean',
                                                                 'num_50_mean',
                                                                 'num_75_mean',
                                                                 'num_985_mean',
                                                                 'num_100_mean',
                                                                 'num_unq_mean',
                                                                 'total_secs_mean',
                                                                 'num_25_sum',
                                                                 'num_50_sum',
                                                                 'num_75_sum',
                                                                 'num_985_sum',
                                                                 'num_100_sum',
                                                                 'num_unq_sum',
                                                                 'total_secs_sum',
                                                                 'num_25_max',
                                                                 'num_50_max',
                                                                 'num_75_max',
                                                                 'num_985_max',
                                                                 'num_100_max',
                                                                 'num_unq_max',
                                                                 'total_secs_max',
                                                                 'subscription_ratio']], axis=0)

    train_logs_normalized = data[0:len(train_logs)]
    test_logs_normalized = data[len(train_logs):]

    train_logs_normalized.to_csv(
        'new_data/selected2/extra_features/train_static_preprocessed.csv', index=False)
    test_logs_normalized.to_csv(
        'new_data/selected2/extra_features/test_static_preprocessed.csv', index=False)
    return train_logs_normalized, test_logs_normalized


# preprocess_logs(train_path='new_data/selected2/extra_features/train_logs_extracted_features.csv',
#                 test_path='new_data/selected2/extra_features/test_logs_extracted_features.csv')

# preprocess_members(data_path='data/members.csv')

# preprocess_transactions(train_path='new_data/selected2/extra_features/train_transactions_extracted_features.csv',
#                         test_path='new_data/selected2/extra_features/test_transactions_extracted_features.csv')

# preprocess_static_data(train_path='new_data/selected2/extra_features/train_static.csv',
#                        test_path='new_data/selected2/extra_features/test_static.csv')
