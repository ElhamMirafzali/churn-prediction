import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from feature_selection.get_x_y import get_x_y

X, y, _ = get_x_y(df1_path='../balanced_data_split/train_logs.csv',
                  df2_path='../balanced_data_split/train_labels.csv',
                  col='msno')

model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
# plot graph of feature importances for better visualization
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh', color=['b', 'g', 'r', 'c', 'm', 'y', 'g'])
plt.show()
