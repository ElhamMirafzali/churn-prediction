import pandas as pd
import numpy as np
import os
from datetime import date


def get_data_frame(path):
    df = pd.DataFrame()
    for chunk in pd.read_csv(path, chunksize=100000):
        df = df.append(chunk)
    return df


def extract_features_of_logs(input_df, start_date, destination_path):
    users = input_df.msno.unique()
    user_index = 0
    for user in users:
        sequence = input_df.loc[input_df['msno'] == user].sort_values('date').reset_index(drop=True)
        for i in range(len(sequence)):
            if i == 0:
                sequence.loc[i, 'last_log_date_of_25'] = start_date
                sequence.loc[i, 'last_log_date_of_50'] = start_date
                sequence.loc[i, 'last_log_date_of_75'] = start_date
                sequence.loc[i, 'last_log_date_of_985'] = start_date
                sequence.loc[i, 'last_log_date_of_100'] = start_date
                sequence.loc[i, 'days_since_last_log_of_25'] = sequence.loc[i, 'date'] - sequence.loc[
                    i, 'last_log_date_of_25']
                sequence.loc[i, 'days_since_last_log_of_50'] = sequence.loc[i, 'date'] - sequence.loc[
                    i, 'last_log_date_of_50']
                sequence.loc[i, 'days_since_last_log_of_75'] = sequence.loc[i, 'date'] - sequence.loc[
                    i, 'last_log_date_of_75']
                sequence.loc[i, 'days_since_last_log_of_985'] = sequence.loc[i, 'date'] - sequence.loc[
                    i, 'last_log_date_of_985']
                sequence.loc[i, 'days_since_last_log_of_100'] = sequence.loc[i, 'date'] - sequence.loc[
                    i, 'last_log_date_of_100']
            else:
                sequence.loc[i, 'last_log_date_of_25'] = np.where(sequence.loc[i - 1, 'num_25'] > 0,
                                                                  sequence.loc[i - 1, 'date'],
                                                                  sequence.loc[i - 1, 'last_log_date_of_25'])
                sequence.loc[i, 'last_log_date_of_50'] = np.where(sequence.loc[i - 1, 'num_50'] > 0,
                                                                  sequence.loc[i - 1, 'date'],
                                                                  sequence.loc[i - 1, 'last_log_date_of_50'])
                sequence.loc[i, 'last_log_date_of_75'] = np.where(sequence.loc[i - 1, 'num_75'] > 0,
                                                                  sequence.loc[i - 1, 'date'],
                                                                  sequence.loc[i - 1, 'last_log_date_of_75'])
                sequence.loc[i, 'last_log_date_of_985'] = np.where(sequence.loc[i - 1, 'num_985'] > 0,
                                                                   sequence.loc[i - 1, 'date'],
                                                                   sequence.loc[i - 1, 'last_log_date_of_985'])
                sequence.loc[i, 'last_log_date_of_100'] = np.where(sequence.loc[i - 1, 'num_100'] > 0,
                                                                   sequence.loc[i - 1, 'date'],
                                                                   sequence.loc[i - 1, 'last_log_date_of_100'])
                sequence.loc[i, 'days_since_last_log_of_25'] = sequence.loc[i, 'date'] - sequence.loc[
                    i, 'last_log_date_of_25']
                sequence.loc[i, 'days_since_last_log_of_50'] = sequence.loc[i, 'date'] - sequence.loc[
                    i, 'last_log_date_of_50']
                sequence.loc[i, 'days_since_last_log_of_75'] = sequence.loc[i, 'date'] - sequence.loc[
                    i, 'last_log_date_of_75']
                sequence.loc[i, 'days_since_last_log_of_985'] = sequence.loc[i, 'date'] - sequence.loc[
                    i, 'last_log_date_of_985']
                sequence.loc[i, 'days_since_last_log_of_100'] = sequence.loc[i, 'date'] - sequence.loc[
                    i, 'last_log_date_of_100']

        # remove temporal dates
        sequence = sequence.drop(
            columns=['last_log_date_of_25', 'last_log_date_of_50', 'last_log_date_of_75', 'last_log_date_of_985',
                     'last_log_date_of_100'])

        if not os.path.isfile(destination_path):
            sequence.to_csv(destination_path, index=False)
        else:
            sequence.to_csv(destination_path, mode='a', index=False, header=False)

        user_index += 1
        if user_index % 100 == 0:
            print('Number of completed users = %d' % user_index)


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

    return dataset


def extract_features_of_transactions(input_df, destination_path):
    users = input_df.msno.unique()
    user_index = 0
    for user in users:
        sequence = input_df.loc[input_df['msno'] == user].sort_values('transaction_date').reset_index(drop=True)
        trans_data = avg_time_between_trans(sequence)
        if not os.path.isfile(destination_path):
            trans_data.to_csv(destination_path, index=False)
        else:
            trans_data.to_csv(destination_path, mode='a', index=False, header=False)

        user_index += 1
        if user_index % 500 == 0:
            print('Number of completed users = %d' % user_index)

# logs = 'new_data/selected2/test_logs.csv'
# data_frame: pd.DataFrame = get_data_frame(logs)
# print("Data frame is ready.")

# extract_features_of_logs(input_df=data_frame, start_date=20170301,
#                  destination_path='new_data/selected2/test_logs_with_extracted_features.csv')


# transactions_path = 'new_data/selected2/test_transactions.csv'
# data_frame: pd.DataFrame = get_data_frame(transactions_path)
# print("Data frame is ready.")
#
# extract_features_of_transactions(input_df=data_frame,
#                                  destination_path='new_data/selected2/test_transactions_with_extracted_features.csv')
