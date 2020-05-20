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
        if user_index % 500 == 0:
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


def extract_avg_time_between_trans(input_df, destination_path):
    users = input_df.msno.unique()
    user_index = 0
    for user in users:
        sequence = input_df.loc[input_df['msno'] == user].sort_values('transaction_date').reset_index(drop=True)

        # average time between transactions
        trans_data = avg_time_between_trans(sequence)
        if not os.path.isfile(destination_path):
            trans_data.to_csv(destination_path, index=False)
        else:
            trans_data.to_csv(destination_path, mode='a', index=False, header=False)

        user_index += 1
        if user_index % 500 == 0:
            print('Number of completed users = %d' % user_index)


def subtract_dates(df1, col1, df2, col2, i):
    date1 = str(df1.loc[i, col1])
    date2 = str(df2.loc[0, col2])
    date1 = date(year=int(date1[0:4]), month=int(date1[4:6]), day=int(date1[6:8]))
    date2 = date(year=int(date2[0:4]), month=int(date2[4:6]), day=int(date2[6:8]))
    return (date1 - date2).days


def extract_days_since_registration(trans_df, members_df, logs_df):
    # transactions
    users = trans_df.msno.unique()
    new_trans_df = pd.DataFrame()
    for user in users:
        trans_sequence = trans_df.loc[trans_df['msno'] == user].sort_values('transaction_date').reset_index(drop=True)
        member_data = members_df.loc[members_df['msno'] == user].reset_index(drop=True)
        for i in range(len(trans_sequence)):
            date1 = str(trans_sequence.loc[i, 'transaction_date'])
            date2 = str(member_data.loc[0, 'registration_init_time'])
            date1 = date(year=int(date1[0:4]), month=int(date1[4:6]), day=int(date1[6:8]))
            date2 = date(year=int(date2[0:4]), month=int(date2[4:6]), day=int(date2[6:8]))

            trans_sequence.loc[i, 'days_since_registration'] = (date1 - date2).days

        new_trans_df = new_trans_df.append(trans_sequence, ignore_index=True)

    # logs
    users = logs_df.msno.unique()
    new_logs_df = pd.DataFrame()
    for user in users:
        logs_sequence = logs_df.loc[logs_df['msno'] == user].sort_values('date').reset_index(drop=True)
        member_data = members_df.loc[members_df['msno'] == user].reset_index(drop=True)
        for i in range(len(logs_sequence)):
            date1 = str(logs_sequence.loc[i, 'date'])
            date2 = str(member_data.loc[0, 'registration_init_time'])
            date1 = date(year=int(date1[0:4]), month=int(date1[4:6]), day=int(date1[6:8]))
            date2 = date(year=int(date2[0:4]), month=int(date2[4:6]), day=int(date2[6:8]))

            logs_sequence.loc[i, 'days_since_registration'] = (date1 - date2).days

        new_logs_df = new_logs_df.append(logs_sequence, ignore_index=True)

    return new_trans_df, new_logs_df


def extract_days_since_last_cancellation(trans_df):
    users = trans_df.msno.unique()
    for user in users:
        sequence = trans_df.loc[trans_df['msno'] == user].sort_values('transaction_date').reset_index(drop=True)


# I think no need!
def total_num_of_cancellations(input_df):
    users = input_df.msno.unique()
    user_index = 0
    for user in users:
        sequence = input_df.loc[input_df['msno'] == user].sort_values('transaction_date').reset_index(drop=True)
        df = pd.DataFrame([{'msno': user}])
        df['total_cancellations'] = sequence['is_cancel'].sum()


def extract_static_data_of_logs(input_df, destination_path):
    users = input_df.msno.unique()
    user_index = 0
    for user in users:
        sequence = input_df.loc[input_df['msno'] == user].sort_values('date').reset_index(drop=True)
        df = pd.DataFrame([{'msno': user}])
        df['num_25_mean'] = sequence['num_25'].mean()
        df['num_50_mean'] = sequence['num_50'].mean()
        df['num_75_mean'] = sequence['num_75'].mean()
        df['num_985_mean'] = sequence['num_985'].mean()
        df['num_100_mean'] = sequence['num_100'].mean()
        df['num_unq_mean'] = sequence['num_unq'].mean()
        df['total_secs_mean'] = sequence['total_secs'].mean()

        df['num_25_sum'] = sequence['num_25'].sum()
        df['num_50_sum'] = sequence['num_50'].sum()
        df['num_75_sum'] = sequence['num_75'].sum()
        df['num_985_sum'] = sequence['num_985'].sum()
        df['num_100_sum'] = sequence['num_100'].sum()
        df['num_unq_sum'] = sequence['num_unq'].sum()
        df['total_secs_sum'] = sequence['total_secs'].sum()

        df['num_25_max'] = sequence['num_25'].max()
        df['num_50_max'] = sequence['num_50'].max()
        df['num_75_max'] = sequence['num_75'].max()
        df['num_985_max'] = sequence['num_985'].max()
        df['num_100_max'] = sequence['num_100'].max()
        df['num_unq_max'] = sequence['num_unq'].max()
        df['total_secs_max'] = sequence['total_secs'].max()

        if not os.path.isfile(destination_path):
            df.to_csv(destination_path, index=False)
        else:
            df.to_csv(destination_path, mode='a', index=False, header=False)

        user_index += 1
        if user_index % 500 == 0:
            print('Number of completed users = %d' % user_index)


# logs

logs = 'new_data/selected2/train_logs.csv'
logs_data_frame: pd.DataFrame = get_data_frame(logs)
print("Logs data frame is ready.")

# extract_features_of_logs(input_df=data_frame, start_date=20170301,
#                  destination_path='new_data/selected2/test_logs_with_extracted_features.csv')

# extract_static_data_of_logs(data_frame, destination_path='new_data/selected2/test_logs_static_data.csv')

# transactions

transactions_path = 'new_data/selected2/train_transactions.csv'
trans_data_frame: pd.DataFrame = get_data_frame(transactions_path)
print("Transactions data frame is ready.")

members_path = 'new_data/selected2/train_members.csv'
members_data_frame = get_data_frame(members_path)
print("Members data frame is ready.")

# extract_avg_time_between_trans(input_df=data_frame,
#                                  destination_path='new_data/selected2/test_transactions_with_extracted_features.csv')


# total_num_of_cancellations(data_frame)

# new_trans_data_frame, new_logs_data_frame = extract_days_since_registration(trans_df=trans_data_frame,
#                                                                             members_df=members_data_frame,
#                                                                             logs_df=logs_data_frame)
