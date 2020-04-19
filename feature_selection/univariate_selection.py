import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from feature_selection.get_x_y import get_x_y

X, y, _ = get_x_y(df1_path='../balanced_data_split/train_logs.csv',
                  df2_path='../balanced_data_split/train_labels.csv',
                  col='msno')

# apply SelectKBest class to score the features
best_features = SelectKBest(score_func=chi2, k='all')
fit = best_features.fit(X, y)
scores = pd.DataFrame(fit.scores_)
columns = pd.DataFrame(X.columns)
# concat two data frames for better visualization
featureScores = pd.concat([columns, scores], axis=1)
featureScores.columns = ['Feature', 'Score']  # naming the data frame columns
print(featureScores.nlargest(10, 'Score'))  # print 10 best features
