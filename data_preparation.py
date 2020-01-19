import pandas as pd
from sklearn.model_selection import train_test_split


def reduce_labels_dataset(csv_path, churn_sample_fraction, non_churn_sample_fraction):
    # csv_path='data/february_labels.csv'
    labels = pd.read_csv(csv_path)
    churn = labels[labels['is_churn'] == 1]
    non_churn = labels[labels['is_churn'] == 0]
    churn_sample = churn.sample(frac=churn_sample_fraction)
    non_churn_sample = non_churn.sample(frac=non_churn_sample_fraction)
    labels_reduced = pd.concat([churn_sample, non_churn_sample], ignore_index=True)
    return labels_reduced


def find_transactions(transactions, users):
    new_transactions = common_column(transactions, users, 'msno')
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
# targets = reduce_labels_dataset(csv_path='data/february_labels.csv',
#                                  churn_sample_fraction=0.8, non_churn_sample_fraction=0.05)
# targets.to_csv(path_or_buf='data/february_labels_reduced.csv', index=False)


# ------------------------ to find transactions of users in the reduced labels dataset
# l = pd.read_csv('data/february_labels_reduced.csv')
# t = pd.read_csv('data/transactions_in_january.csv')
# t = find_transactions(transactions=t, users=l)
# t.to_csv(path_or_buf='data/transactions_in_january_reduced.csv', index=False)


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
users = pd.read_csv('data/february_labels_reduced_edited.csv')
trans = pd.read_csv('data/transactions_in_january_reduced_edited.csv')
train_users, test_users = train_test_split(users, train_size=0.6)
train_transactions = find_transactions(transactions=trans, users=train_users)
test_transactions = find_transactions(transactions=trans, users=test_users)

train_transactions.to_csv('data_split/train_transactions.csv', index=False)
test_transactions.to_csv('data_split/test_transactions.csv', index=False)
train_users.to_csv('data_split/train_labels.csv', index=False)
test_users.to_csv('data_split/test_labels.csv', index=False)

print("train_transactions = \n", train_transactions, '\n')
print("test_transactions = \n", test_transactions)
