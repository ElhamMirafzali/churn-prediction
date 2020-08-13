import seaborn as sns
import matplotlib.pyplot as plt
from feature_selection.get_x_y import get_x_y

X, y, data = get_x_y(df1_path='../new_data/selected2/train_logs_with_extracted_features.csv',
                     df2_path='../new_data/selected2/train_labels.csv',
                     col='msno')

# Correlation Matrix with Heatmap
# Correlation states how the features are related to each other or the target variable

# get correlations of each features in dataset
correlation_matrix = data.corr()
top_corr_features = correlation_matrix.index
plt.figure(figsize=(12, 10))
# plot heat map
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()
