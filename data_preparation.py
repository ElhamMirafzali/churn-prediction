import pandas as pd


def reduce_labels_dataset(csv_path):
    # csv_path='data/february_labels.csv'
    labels = pd.read_csv(csv_path)
    churn = labels[labels['is_churn'] == 1]
    non_churn = labels[labels['is_churn'] == 0]
    churn_sample = churn.sample(frac=0.8)
    non_churn_sample = non_churn.sample(frac=0.05)
    labels_reduced = pd.concat([churn_sample, non_churn_sample], ignore_index=True)
    return labels_reduced


def find_transactions(labels_path, transactions_path):
    labels = pd.read_csv(labels_path)
    transactions = pd.read_csv(transactions_path)
    i1 = labels.set_index('msno').index
    i2 = transactions.set_index('msno').index
    new_transactions = transactions[i2.isin(i1)]
    return new_transactions


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


def transactions_in_range(trans_path, lower_bound, upper_bound):
    trans = pd.read_csv(trans_path)
    trans_in_range = trans[(trans['membership_expire_date'] >= lower_bound) &
                           (trans['membership_expire_date'] <= upper_bound)]
    # trans_in_range.to_csv('data/transactions_in_march.csv', index=False)
    return trans_in_range

#####################################################################

# ------------------------ to reduce the labels dataset
# targets = reduce_labels_dataset(csv_path='data/february_labels.csv')
# targets.to_csv(path_or_buf='data/february_labels_reduced.csv', index=False)


# ------------------------ to find transactions of users in the reduced labels dataset
# t = find_transactions(labels_path='data/february_labels_reduced.csv',
#                       transactions_path='data/transactions_in_january.csv')
# t.to_csv(path_or_buf='data/transactions_in_january_reduced.csv', index=False)


# ------------------------ to find users with transactions count more than a limit
# u, tr, lb = find_users_with_transaction_limit(labels_path='data/february_labels.csv',
#                                               transactions_path='data/transactions_in_january.csv',
#                                               transactions_count_limit=2)
# t_reduced = pd.read_csv('data/transactions_in_january_reduced.csv')
# l_reduced = pd.read_csv('data/february_labels_reduced.csv')
# tr_concat = pd.concat([tr, t_reduced], ignore_index=True, sort=False)
# lb_concat = pd.concat([lb, l_reduced], ignore_index=True, sort=False)
# tr_concat.to_csv(path_or_buf='data/transactions_in_january_reduced_edited.csv', index=False)
# lb_concat.to_csv(path_or_buf='data/february_labels_reduced_edited.csv', index=False)
