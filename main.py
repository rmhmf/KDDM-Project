import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import preproc


class Data:
    def __init__(self):
        self.dir = "data/data_6.csv"
        self.data = self.load_data()

    def load_data(self):
        df = pd.read_csv(self.dir)
        return df

    def save_data(self, text):
        self.data.to_csv(text, index=False)

    def get_data(self):
        return self.data

    def data_desc(self):
        print(self.data.info())
        print(self.data.describe())
        print(self.data.head())


class EDA:
    def __init__(self, data):
        self.data = data

    def eda_report(self, col):
        self.statistic_print(col)
        self.visual_missing_value(col)
        self.plot_boxplot(col)
        self.plot_count_bar(col)

    def statistic_print(self, col):
        print(self.data[col].describe())

    def plot_corr(self):
        numerical_data = self.data.select_dtypes(include=['float64', 'Int32', 'Float64'])
        correlation_matrix = numerical_data.corr().round(3)
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.tight_layout(pad=2.0)
        plt.title('Correlation Matrix')
        plt.show()

    def visual_missing_value(self, col=None):
        plt.figure(figsize=(12, 10))
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
        plt.figure(figsize=(12, 10))
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
        plt.figure(figsize=(12, 10))
        if self.data[col].dtype == float:
            bins = pd.cut(self.data[col], bins=10)
            print(bins.value_counts())
            sns.countplot(x=bins)
        else:
            print(self.data[col].value_counts())
            sns.countplot(data=self.data, x=col)
        plt.title(f"Bar chart of column {col}", fontsize=18)
        plt.xticks(fontsize=24)
        plt.ylabel('Values')
        plt.tight_layout(pad=2.0)
        plt.show()

    def statistic_print(self, col1, col2):
        print(self.data.groupby(col1)[col2].agg(['mean', 'std']))

    def plot_barplot(self, col1, col2):
        plt.figure(figsize=(12, 10))
        sns.barplot(x=col1, y=col2, data=self.data, color='skyblue')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title('Bar Plot')
        plt.show()

    def plot_scatter(self, col1, col2):
        size_df = self.data.groupby([col1, col2]).size().reset_index(name='counts')
        sns.scatterplot(x=col1, y=col2, data=size_df, size='counts',
                        sizes=(size_df['counts'].min(), size_df['counts'].max()))
        sns.regplot(x=col1, y=col2, data=self.data, scatter=False, color='red')
        plt.title('Scatter plot')
        plt.legend(loc='upper right')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()

data = Data()
data.data_desc()
df = data.get_data()

preproc = preproc.PreProc(df)
data.save_data("preprocV0.1.csv")

eda = EDA(df)
eda.visual_missing_value()
eda.plot_corr()
# col1 = 'Bedrooms'
# col2 = 'Location'
# eda.statistic_print(col1, col2)
# eda.plot_scatter(col1, col2)
# eda.plot_barplot(col1, col2)

# eda.eda_report('SquareFootageHouse')

