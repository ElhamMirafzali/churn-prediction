import pandas as pd
import numpy as np
import os


def get_data_frame(path):
    df = pd.DataFrame()
    for chunk in pd.read_csv(path, chunksize=100000):
        df = df.append(chunk)
    return df


logs = 'new_data/selected2/test_logs.csv'
data_frame: pd.DataFrame = get_data_frame(logs)
print("Data frame is ready.")
users = data_frame.msno.unique()
users_num = len(users)
start_date = 20170201
destination_path = 'new_data/selected2/test_logs_with_extracted_features.csv'
user_index = 0
for user in users:
    sequence = data_frame.loc[data_frame['msno'] == user].sort_values('date').reset_index(drop=True)
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
