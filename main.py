import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Data:

    def __init__(self):
        self.dir = "data/data_6.csv"
        self.data = self.load_data()

    def load_data(self):
        df = pd.read_csv(self.dir)
        return df

    def get_data(self):
        return self.data

    def data_desc(self):
        print(self.data.info())
        print(self.data.describe())
        print(self.data.head())


class PreProc:

    def __init__(self, data):
        self.data = data


class EDA:

    def __init__(self, data):
        self.data = data

    def statistic_print(self, col):
        print(self.data[col].describe())

    def visual_missing_value(self, col=None):
        plt.figure(figsize=(14, 12))
        if col:
            sns.heatmap(self.data[col].isnull().values[:, np.newaxis], cbar=False, cmap='viridis')
            print(f'Number of missing values: {self.data[col].isnull().value_counts()}')
        else:
            sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
        plt.xticks(rotation=60, ha='right', fontsize=14)
        plt.title('Heatmap of Missing Values', fontsize=20)
        plt.tight_layout(pad=2.0)
        plt.show()

    def plot_boxplot(self, col=None):
        plt.figure(figsize=(14, 12))
        if col:
            sns.boxplot(data=self.data, x=col)
        else:
            sns.boxplot(data=self.data)
        plt.title(f'Boxplot of Column(s)')
        plt.xticks(rotation=60, ha='right', fontsize=14)
        plt.ylabel('Values')
        plt.tight_layout(pad=2.0)
        plt.show()

    def plot_count_bar(self, col):
        plt.figure(figsize=(14, 12))
        if self.data[col].dtype == float:
            bins = pd.cut(self.data[col], bins=10)
            print(bins.value_counts())
            sns.countplot(x=bins)
        else:
            sns.countplot(data=self.data, x=col)
        plt.title(f"Bar chart of column {col}", fontsize=18)
        plt.xticks(fontsize=14)
        plt.ylabel('Values')
        plt.tight_layout(pad=2.0)
        plt.show()


data = Data()
data.data_desc()
df = data.get_data()

eda = EDA(df)
col = 'SquareFootageHouse'
eda.statistic_print(col)
eda.visual_missing_value(col)
eda.plot_boxplot(col)
eda.plot_count_bar(col)
