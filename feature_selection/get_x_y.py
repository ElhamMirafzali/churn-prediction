import pandas as pd


def get_x_y(df1_path, df2_path, col):
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    df1_df2_merged = pd.merge(df1, df2, on=col, left_index=True, right_index=True)
    end = len(df1_df2_merged.iloc[0, :])
    x = pd.DataFrame(df1_df2_merged.iloc[:, 1:end - 1])
    y = pd.DataFrame(df1_df2_merged.iloc[:, -1])
    return x, y, df1_df2_merged
