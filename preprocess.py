import pandas as pd
from sklearn import preprocessing


def preprocess_transactions(data_path):
    data = pd.read_csv(data_path)

    # filling missing values with 0
    data = data.fillna(0)

    # normalization
    data[['payment_method_id',
          'payment_plan_days',
          'plan_list_price',
          'actual_amount_paid']] = preprocessing.normalize(data[['payment_method_id',
                                                                 'payment_plan_days',
                                                                 'plan_list_price',
                                                                 'actual_amount_paid']])
    return data
