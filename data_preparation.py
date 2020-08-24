import time
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import os


def print_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)


def reduce_labels_dataset(labels_path, churn_sample_fraction, non_churn_sample_fraction):
    labels = pd.read_csv(labels_path)
    churn = labels[labels['is_churn'] == 1]
    non_churn = labels[labels['is_churn'] == 0]
    churn_sample = churn.sample(frac=churn_sample_fraction)
    non_churn_sample = non_churn.sample(frac=non_churn_sample_fraction)
    labels_reduced = pd.concat([churn_sample, non_churn_sample], ignore_index=True)
    return labels_reduced


def find_transactions(transactions_path, users_path, destination_path):
    users = pd.read_csv(users_path)
    transactions = pd.read_csv(transactions_path)
    new_transactions = common_column(transactions, users, 'msno')
    new_transactions.to_csv(destination_path, index=False)


def find_users_with_transaction_limit(labels_path, transactions_path, transactions_count_limit):
    labels = pd.read_csv(labels_path)
    transactions = pd.read_csv(transactions_path)
    users = transactions.groupby('msno').size().to_frame(name='transactions_count')
    users = users[users['transactions_count'] >= transactions_count_limit]
    users['msno'] = users.index
    new_labels = common_column(labels, users, 'msno')
    new_transactions = common_column(transactions, new_labels, 'msno')
    return users, new_transactions, new_labels


def common_column(df1, df2, col):
    i1 = df1.set_index(col).index
    i2 = df2.set_index(col).index
    new_df = df1[i1.isin(i2)]
    return new_df


def transactions_in_range(source_path, lower_bound, upper_bound, destination_path):
    for chunk in pd.read_csv(source_path, chunksize=100000):
        trans_in_range: pd.DataFrame = chunk[(chunk['membership_expire_date'] >= lower_bound) &
                                             (chunk['membership_expire_date'] <= upper_bound)]
        if not os.path.isfile(destination_path):
            trans_in_range.to_csv(destination_path, index=False)
        else:
            trans_in_range.to_csv(destination_path, mode='a', index=False, header=False)


#####################################################################
# LOGS
#####################################################################
def select_logs_in_range(source_path, lower_bound, upper_bound, destination_path):
    for chunk in pd.read_csv(source_path, chunksize=100000):
        logs_in_range: pd.DataFrame = chunk[(chunk['date'] >= lower_bound) &
                                            (chunk['date'] <= upper_bound)]
        if not os.path.isfile(destination_path):
            logs_in_range.to_csv(destination_path, index=False)
        else:
            logs_in_range.to_csv(destination_path, mode='a', index=False, header=False)


def find_logs(logs_path, users_path, destination_path):
    users = pd.read_csv(users_path)
    for chunk in pd.read_csv(logs_path, chunksize=100000):
        selected_logs = common_column(chunk, users, 'msno')
        if not os.path.isfile(destination_path):
            selected_logs.to_csv(destination_path, index=False)
        else:
            selected_logs.to_csv(destination_path, mode='a', index=False, header=False)


def find_users_with_logs(users_path, logs_path, destination_path):
    users = pd.read_csv(users_path)
    logs = pd.read_csv(logs_path)
    selected_users = common_column(users, logs, 'msno')
    selected_users.to_csv(destination_path, index=False)


#####################################################################
# MEMBERS
#####################################################################
def find_members(members_path, users_path, destination_path):
    users = pd.read_csv(users_path)
    for chunk in pd.read_csv(members_path, chunksize=100000):
        selected_members = common_column(chunk, users, 'msno')
        if not os.path.isfile(destination_path):
            selected_members.to_csv(destination_path, index=False)
        else:
            selected_members.to_csv(destination_path, mode='a', index=False, header=False)


#####################################################################
# MEMBERS
#####################################################################

# ------------------------ find members data

# find_members(members_path='data/members_preprocessed.csv', users_path='new_data/selected2/test_labels.csv',
#              destination_path='new_data/selected2/test_members.csv')
# the reverse is necessary
# find_members(source_path='new_data/selected2/test_labels.csv', users_path='new_data/selected2/test_members.csv',
#              destination_path='new_data/selected2/test_labels_2.csv')

# find_members(members_path='data/members_preprocessed.csv', users_path='new_data/selected2/train_labels.csv',
#              destination_path='new_data/selected2/train_members.csv')

#####################################################################
# STATIC DATA
#####################################################################
# train_members = pd.read_csv('new_data/selected2/test_members.csv')
# train_static_trans = pd.read_csv('new_data/selected2/extra_features/test_transactions_static.csv')
# train_static_logs = pd.read_csv('new_data/selected2/extra_features/test_logs_static.csv')
#
# train_static = pd.merge(train_members, train_static_logs, on='msno')
# train_static = pd.merge(train_static, train_static_trans, on='msno')
# train_static.to_csv('new_data/selected2/extra_features/test_static.csv', index=False)
#####################################################################

# ------------------------ combine two transactions data

# transactions_path = 'data1/transactions.csv'
# for chunk in pd.read_csv(transactions_path, chunksize=100000):
#     chunk.to_csv('data/transactions_all.csv', mode='a', index=False, header=False)

############################################################
# ------------------------ find transactions in range

# transactions_in_range(source_path='data/transactions_all.csv', lower_bound=20170201,
#                       upper_bound=20170228,
#                       destination_path='data/transactions(membership_expires_in_feb).csv')

# transactions_in_range(source_path='data/transactions_all.csv', lower_bound=20170301,
#                       upper_bound=20170331,
#                       destination_path='data/transactions(membership_expires_in_march).csv')

############################################################
# ------------------------ intersect of train.csv and transactions(membership_expires_in_feb).csv and logs_jan.csv
# train_total_labels = pd.read_csv('data/train.csv')
# trans_membership_expires_in_jan = pd.read_csv('new_data/transactions(membership_expires_in_feb).csv')
# train_labels = common_column(df1=train_total_labels, df2=trans_membership_expires_in_jan, col="msno")
# logs = pd.read_csv('new_data/logs_feb.csv')
# train_labels = common_column(df1=train_labels, df2=logs, col="msno")
# train_labels.to_csv('new_data/train_labels_intersect.csv', index=False)

# ------------------------ intersect of test.csv and transactions(membership_expires_in_march).csv and logs_feb.csv
# test_total_labels = pd.read_csv('new_data/test.csv')
# trans_membership_expires_in_march = pd.read_csv('new_data/transactions(membership_expires_in_march).csv')
# logs = pd.read_csv('new_data/logs_march.csv')
# test_labels = common_column(df1=test_total_labels, df2=trans_membership_expires_in_march, col="msno")
# test_labels = common_column(df1=test_labels, df2=logs, col="msno")
# test_labels.to_csv('new_data/test_labels_intersect.csv', index=False)

#####################################################################

# ------------------------ to reduce the labels dataset
# targets = reduce_labels_dataset(labels_path='new_data/test_labels_intersect.csv',
#                                 churn_sample_fraction=0.3019, non_churn_sample_fraction=0.0192)
# targets.to_csv(path_or_buf='new_data/selected2/test_labels.csv', index=False)

#####################################################################
# Transactions
#
#####################################################################

# ------------------------ to find transactions of users
# users = pd.read_csv('new_data/selected2/test_labels.csv')
# transactions = pd.read_csv('new_data/transactions(membership_expires_in_march).csv')
# transactions_of_users = find_transactions(transactions=transactions, users=users)
# transactions_of_users.to_csv(path_or_buf='new_data/selected2/test_transactions.csv', index=False)

#####################################################################
# ------------------------ to find users with transactions count more than a limit
# u, tr, lb = find_users_with_transaction_limit(labels_path='data/february_labels.csv',
#                                               transactions_path='data/transactions_in_january.csv',
#                                               transactions_count_limit=2)
# t_reduced = pd.read_csv('data/transactions_in_january_reduced.csv')
# l_reduced = pd.read_csv('data/february_labels_reduced.csv')
# tr_concat = (pd.concat([tr, t_reduced], ignore_index=True, sort=False)).drop_duplicates(keep='first', inplace=False)
# lb_concat = (pd.concat([lb, l_reduced], ignore_index=True, sort=False)).drop_duplicates(keep='first', inplace=False)
# tr_concat.to_csv(path_or_buf='data/transactions_in_january_reduced_edited.csv', index=False)
# lb_concat.to_csv(path_or_buf='data/february_labels_reduced_edited.csv', index=False)


# ------------------------ split train test
# users = pd.read_csv('data/february_labels_reduced_edited.csv')
# trans = pd.read_csv('data/transactions_in_january_reduced_edited.csv')
# train_users, test_users = train_test_split(users, train_size=0.6)
# train_transactions = find_transactions(transactions=trans, users=train_users)
# test_transactions = find_transactions(transactions=trans, users=test_users)
#
# train_transactions.to_csv('data_split/train_transactions.csv', index=False)
# test_transactions.to_csv('data_split/test_transactions.csv', index=False)
# train_users.to_csv('data_split/train_labels.csv', index=False)
# test_users.to_csv('data_split/test_labels.csv', index=False)


# --------------------------

# transactions = pd.read_csv('data/transactions.csv')
# transactions_with_expire_date_in_august_to_january = transactions[
#    (transactions['membership_expire_date'] >= 20160801) &
#    (transactions['membership_expire_date'] < 20170201)]
# transactions_with_expire_date_in_august_to_january.to_csv('data/transactions_with_expire_date_in_august_to_january.csv',
#                                                          index=False)
# print(transactions_with_expire_date_in_august_to_january.shape)


# -------------------------- split train test with transactions more than a limit

# u, tr, lb = find_users_with_transaction_limit(labels_path='data/february_labels.csv',
#                                               transactions_path='data/transactions_with_expire_date_in_august_to_january.csv',
#                                               transactions_count_limit=10)
# train_users, test_users = train_test_split(lb, train_size=0.6)
# train_transactions = find_transactions(transactions=tr, users=train_users)
# test_transactions = find_transactions(transactions=tr, users=test_users)
#
# train_transactions.to_csv('data_split/train_transactions.csv', index=False)
# test_transactions.to_csv('data_split/test_transactions.csv', index=False)
# train_users.to_csv('data_split/train_labels.csv', index=False)
# test_users.to_csv('data_split/test_labels.csv', index=False)


# -------------------------------------------

# transactions = pd.read_csv('data/transactions_with_expire_date_in_august_to_january.csv')
# labels_of_feb = pd.read_csv('data/february_labels.csv')
# churners = labels_of_feb[labels_of_feb['is_churn'] == 1]
# trans_of_churners = find_transactions(transactions=transactions, users=churners)
# u, tr, lb = find_users_with_transaction_limit(labels_path='data/february_labels.csv',
#                                               transactions_path='data/transactions_with_expire_date_in_august_to_january.csv',
#                                               transactions_count_limit=10)
#
# tr_concat = (pd.concat([tr, trans_of_churners], ignore_index=True, sort=False)).drop_duplicates(keep='first',
#                                                                                                 inplace=False)
# lb_concat = (pd.concat([lb, churners], ignore_index=True, sort=False)).drop_duplicates(keep='first', inplace=False)
#
# train_users, test_users = train_test_split(lb_concat, train_size=0.6)
# train_transactions = find_transactions(transactions=tr_concat, users=train_users)
# test_transactions = find_transactions(transactions=tr_concat, users=test_users)
#
# train_transactions.to_csv('data_split/train_transactions.csv', index=False)
# test_transactions.to_csv('data_split/test_transactions.csv', index=False)
# train_users.to_csv('data_split/train_labels.csv', index=False)
# test_users.to_csv('data_split/test_labels.csv', index=False)
#
# print("train_users = ", train_users.shape)
# print("test_users = ", test_users.shape)
# print("train_transactions = ", train_transactions.shape)
# print("test_transactions = ", test_transactions.shape)

# ---------------------------------
#
# transactions = pd.read_csv('data/transactions_with_expire_date_in_august_to_january.csv')
# labels_of_feb = pd.read_csv('data/february_labels.csv')
# tr = find_transactions(transactions=transactions, users=labels_of_feb)
# print("tr = ", tr.shape)
# train_users, test_users = train_test_split(labels_of_feb, train_size=0.6)
# train_users = train_users.sample(frac=0.18)
# test_users = test_users.sample(frac=0.18)
# train_transactions = find_transactions(transactions=tr, users=train_users)
# test_transactions = find_transactions(transactions=tr, users=test_users)
#
# train_transactions.to_csv('data_split/train_transactions.csv', index=False)
# test_transactions.to_csv('data_split/test_transactions.csv', index=False)
# train_users.to_csv('data_split/train_labels.csv', index=False)
# test_users.to_csv('data_split/test_labels.csv', index=False)
#
# print("train_users = ", train_users.shape)
# print("test_users = ", test_users.shape)
# print("train_transactions = ", train_transactions.shape)
# print("test_transactions = ", test_transactions.shape)

# ---------------------------------
#
# balanced-data-split
# train_transactions = pd.read_csv('data_split/train_transactions.csv')
# train_users = pd.read_csv('data_split/train_labels.csv')
# test_transactions = pd.read_csv('data_split/test_transactions.csv')
# test_users = pd.read_csv('data_split/test_labels.csv')
#
# # for train data
# percentage = train_users['is_churn'].value_counts(normalize=True) * 100
# print("percentage = ", percentage)
#
# churners = train_users[train_users['is_churn'] == 1]
# trans_of_churners = find_transactions(transactions=train_transactions, users=churners)
#
# non_churners = train_users[train_users['is_churn'] == 0]
# non_churners = non_churners.sample(frac=0.06)
# trans_of_non_churners = find_transactions(transactions=train_transactions, users=non_churners)
#
# train_transactions = (
#     pd.concat([trans_of_churners, trans_of_non_churners], ignore_index=True, sort=False)).drop_duplicates(
#     keep='first',
#     inplace=False)
# train_users = (pd.concat([churners, non_churners], ignore_index=True, sort=False)).drop_duplicates(keep='first',
#                                                                                                    inplace=False)
# percentage = train_users['is_churn'].value_counts(normalize=True) * 100
# print("percentage = ", percentage)
#
# train_transactions.to_csv('balanced_data_split/train_transactions.csv', index=False)
# train_users.to_csv('balanced_data_split/train_labels.csv', index=False)
#
# # for test data
# percentage = test_users['is_churn'].value_counts(normalize=True) * 100
# print("percentage = ", percentage)
#
# churners = test_users[test_users['is_churn'] == 1]
# trans_of_churners = find_transactions(transactions=test_transactions, users=churners)
#
# non_churners = test_users[test_users['is_churn'] == 0]
# non_churners = non_churners.sample(frac=0.06)
# trans_of_non_churners = find_transactions(transactions=test_transactions, users=non_churners)
#
# test_transactions = (
#     pd.concat([trans_of_churners, trans_of_non_churners], ignore_index=True, sort=False)).drop_duplicates(
#     keep='first',
#     inplace=False)
# test_users = (pd.concat([churners, non_churners], ignore_index=True, sort=False)).drop_duplicates(keep='first',
#                                                                                                   inplace=False)
# percentage = test_users['is_churn'].value_counts(normalize=True) * 100
# print("percentage = ", percentage)
# test_transactions.to_csv('balanced_data_split/test_transactions.csv', index=False)
# test_users.to_csv('balanced_data_split/test_labels.csv', index=False)
#
# print("train_users = ", train_users.shape)
# print("test_users = ", test_users.shape)
# print("train_transactions = ", train_transactions.shape)
# print("test_transactions = ", test_transactions.shape)

# ##########################################################
# LOGS
#
############################################################

# ------------------------ combine two transactions data

# logs_path = 'data1/user_logs.csv'
# for chunk in pd.read_csv(logs_path, chunksize=100000):
#     chunk.to_csv('new_data/user_logs_all.csv', mode='a', index=False, header=False)

#############################################################

# select_logs_in_range(source_path='data/user_logs_all.csv', lower_bound=20170101, upper_bound=20170331,
#                      destination_path='data/test_logs_jan2017_to_march2017.csv')
# select_logs_in_range(source_path='new_data/user_logs_all.csv', lower_bound=20170101, upper_bound=20170131,
#                      destination_path='new_data/logs_jan.csv')
# select_logs_in_range(source_path='new_data/user_logs_all.csv', lower_bound=20170201, upper_bound=20170228,
#                      destination_path='new_data/logs_feb.csv')
# select_logs_in_range(source_path='new_data/user_logs_all.csv', lower_bound=20170301, upper_bound=20170331,
#                      destination_path='new_data/logs_march.csv')

############################################################
# t = pd.read_csv('new_data/selected/train_labels.csv')
# l = pd.read_csv('new_data/logs_feb.csv')
# train_labels = common_column(df1=t, df2=l, col="msno")
# train_labels.to_csv('new_data/selected/test/train_labels.csv')
############################################################
# -------------------- to find logs of users
# find_logs(users_path='new_data/selected2/train_labels.csv',
#                 logs_path='new_data/logs_feb.csv',
#                 destination_path='new_data/selected2/train_logs.csv')
# find_logs(users_path='new_data/selected2/test_labels.csv',
#                 logs_path='new_data/logs_march.csv',
#                 destination_path='new_data/selected2/test_logs.csv')

############################################################

# find_users_with_logs(users_path='balanced_data_split/train_labels.csv',
#                      logs_path='balanced_data_split/train_logs.csv',
#                      destination_path='balanced_data_split/train_labels_selected.csv')

############################################################
# static = pd.read_csv('new_data/selected2/extra_features/train_static_preprocessed.csv')
# static = static[static.columns.difference(['num_25_sum',
#                                            'num_50_sum',
#                                            'num_75_sum',
#                                            'num_985_sum',
#                                            'num_100_sum',
#                                            'num_unq_sum',
#                                            'total_secs_sum',
#                                            'num_25_max',
#                                            'num_50_max',
#                                            'num_75_max',
#                                            'num_985_max',
#                                            'num_100_max',
#                                            'num_unq_max',
#                                            'total_secs_max'])]
#
# static.to_csv('new_data/selected2/extra_features/train_static_preprocessed_reduced.csv', index=False)

################################################
# make data ready for colab
################################################

################################################
# members age filtering
################################################

# data = pd.read_csv('data/all_raw_data/members.csv')
# print(data.shape)
# data = data[data.bd <= 120]
# data = data[data.bd >= 0]
# print(data.shape)
# data.to_csv('data/all_raw_data/members_age_0_120.csv', index=False)

################################################
# intersect of train.csv, members, transactions, logs and update train.csv
################################################

# user_labels = pd.read_csv('data/all_raw_data/test.csv')
# members = pd.read_csv('data/all_raw_data/members_age_0_120.csv')
# user_labels = common_column(df1=user_labels, df2=members, col="msno")
#
# transactions_path = 'data/all_raw_data/transactions(membership_expires_in_march).csv'
# trans_df = pd.DataFrame()
# for chunk in pd.read_csv(transactions_path, chunksize=100000):
#     trans_df = trans_df.append(other=chunk)
# user_labels = common_column(df1=user_labels, df2=trans_df, col="msno")
# user_labels.to_csv('data/test_selected/test_labels_intersect_tmp.csv', index=False)

# print_time()  # 19:24:27
# user_labels = pd.read_csv('data/test_selected/test_labels_intersect_tmp.csv')
# logs_path = 'data/all_raw_data/test_logs_jan2017_to_march2017.csv'
# logs_df = pd.DataFrame()
# for chunk in pd.read_csv(logs_path, chunksize=100000):
#     users = pd.DataFrame()
#     users['msno'] = chunk.msno.unique()
#     if not os.path.isfile('data/test_selected/unique_users_of_test_logs_tmp.csv'):
#         users.to_csv('data/test_selected/unique_users_of_test_logs_tmp.csv', index=False)
#     else:
#         users.to_csv('data/test_selected/unique_users_of_test_logs_tmp.csv', mode='a', index=False, header=False)
#
# print_time()  # 19:27:08

# users = pd.read_csv('data/test_selected/unique_users_of_test_logs_tmp.csv')
# unique_users = pd.DataFrame()
# unique_users['msno'] = users.msno.unique()
# unique_users.to_csv('data/test_selected/unique_users_of_test_logs.csv', index=False)

# user_labels = pd.read_csv('data/test_selected/test_labels_intersect_tmp.csv')
# unique_users_in_logs = pd.read_csv('data/test_selected/unique_users_of_test_logs.csv')
# user_labels = common_column(df1=user_labels, df2=unique_users_in_logs, col="msno")
# user_labels.to_csv('data/test_selected/test_labels_intersect.csv', index=False)

################################################
# ratio of non_churn & churn (60%, 40%) for train & (70%, 30%) for test
################################################

# targets = reduce_labels_dataset(labels_path='data/train_selected/train_labels_intersect.csv',
#                                 churn_sample_fraction=0.6599, non_churn_sample_fraction=0.0367)
# targets.to_csv(path_or_buf='data/train_selected/train_labels_selected.csv', index=False)


# targets = reduce_labels_dataset(labels_path='data/test_selected/test_labels_intersect.csv',
#                                 churn_sample_fraction=0.2342, non_churn_sample_fraction=0.0198)
# targets.to_csv(path_or_buf='data/test_selected/test_labels_selected.csv', index=False)

################################################
# find members, transactions, logs  of users
################################################

users_path = 'data/test_selected/test_labels_selected_plus_trans_count_above_3.csv'
members_path = 'data/all_raw_data/members_age_0_120.csv'
transactions_path = 'data/all_raw_data/transactions(membership_expires_in_march).csv'
logs_path = 'data/all_raw_data/test_logs_jan2017_to_march2017.csv'
# find_members(members_path=members_path,
#              users_path=users_path,
#              destination_path='data/test_selected/test_members.csv')

# find_transactions(transactions_path=transactions_path,
#                   users_path=users_path,
#                   destination_path='data/test_selected/test_transactions(membership_expires_in_march).csv')

find_logs(logs_path=logs_path,
          users_path=users_path,
          destination_path='data/test_selected/test_logs_jan2017_to_march2017.csv')

################################################
# extract users with transactions count more than 3
################################################

# u, tr, lb = find_users_with_transaction_limit(labels_path='data/test_selected/test_labels_intersect.csv',
#                                               transactions_path='data/all_raw_data/transactions(membership_expires_in_march).csv',
#                                               transactions_count_limit=3)

# print(u.shape, tr.shape, lb.shape)
# percentage = lb['is_churn'].value_counts(normalize=True) * 100
# print("percentage = ", percentage)
# lb.to_csv('data/test_selected/test_labels_with_transactions_count_limit_3.csv', index=False)

################################################
# union of two train_labels
################################################

# lb1 = pd.read_csv('data/test_selected/test_labels_selected.csv')
# lb2 = pd.read_csv('data/test_selected/test_labels_with_transactions_count_limit_3.csv')
# merged_lb = pd.concat([lb2, lb1], ignore_index=True, sort=False).drop_duplicates(keep='first', inplace=False)
# merged_lb.to_csv('data/test_selected/test_labels_selected_plus_trans_count_above_3.csv')
#
# print(lb1.shape, lb2.shape, merged_lb.shape)
# users = pd.DataFrame()
# users['msno'] = merged_lb.msno.unique()
# print(users.shape)
# percentage = merged_lb['is_churn'].value_counts(normalize=True) * 100
# print("percentage = ", percentage)

################################################
