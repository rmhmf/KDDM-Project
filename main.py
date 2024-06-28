import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preproc import PreProc
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import missingno as msno
from model import Feature_Sel, LModel, BaseLine


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

# Explainatory Data Analysis
class EDA:
    def __init__(self, data):
        self.data = data

    def eda_report(self, col):
        self.statistic_print(col)
        # self.visual_missing_value(col)
        self.plot_boxplot(col)
        self.plot_count_bar(col)
        self.hist_plot(col)

    def plot_corr(self, data=None):
        numerical_data = self.data.select_dtypes(include=['float64', 'Int32', 'Float64'])
        method = 'pearson' # pearson spearman, kendall
        if data is not None:
            correlation_matrix = data.corr(method=method).round(1)
        else:
            correlation_matrix = numerical_data.corr(method=method).round(1)
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, linecolor='black', vmin=-1, vmax=1)
        plt.xticks(fontsize=6)
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
        plt.xticks(rotation=30, ha='right', fontsize=12)
        plt.title('Heatmap of Missing Values', fontsize=20)
        plt.yticks([])
        plt.tight_layout(pad=5.0)
        plt.show()

    def plot_boxplot(self, col=None):
        plt.figure(figsize=(8, 4))
        if col:
            sns.boxplot(data=self.data, x=col)
        else:
            sns.boxplot(data=self.data)
        plt.title(f'Boxplot of Column(s)')
        plt.xticks(rotation=60, ha='right', fontsize=14)
        plt.ylabel('Values')
        plt.tight_layout(pad=5.0)
        plt.show()

    def plot_count_bar(self, col):
        plt.figure(figsize=(12, 10))
        if self.data[col].dtype == float:
            bins = pd.cut(self.data[col], bins=10)
            print(bins.value_counts())
            sns.countplot(x=bins)
        else:
            # print(self.data[col].value_counts())
            # sns.countplot(data=self.data[col].value_counts)
            value_counts = self.data[col].value_counts().reset_index()
            value_counts.columns = [col, 'count']

            sns.barplot(x=col, y='count', data=value_counts)

        plt.title(f"Bar chart of column {col}", fontsize=18)
        plt.xticks(fontsize=12)
        plt.ylabel('Values')
        plt.tight_layout(pad=2.0)
        plt.show()

    def statistic_print(self, col1, col2=None):
        if col2:
            print(self.data.groupby(col1)[col2].agg(['mean', 'std']))
        else:
            print(self.data[col1].describe())

    def plot_barplot(self, col1, col2):
        plt.figure(figsize=(12, 10))
        sns.barplot(x=col1, y=self.data[col2].astype(float), data=self.data, color='skyblue')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title('Bar Plot')
        plt.show()

    def plot_scatter(self, col1, col2):
        size_df = self.data.groupby([col1, col2]).size().reset_index(name='counts')
        sns.scatterplot(x=col1, y=col2, data=size_df, size='counts',
                        sizes=(size_df['counts'].min(), size_df['counts'].max()))
        # sns.regplot(x=col1, y=col2, data=self.data, scatter=False, color='red')
        plt.title('Scatter plot')
        plt.legend(loc='lower right')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()

    def plot_cat_cat(self, col1, col2):
        plt.figure(figsize=(12, 10))
        sns.countplot(x=col1, hue=col2, data=self.data)
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title('Count Plot')
        plt.show()
    
    def plot_qqplot(self, col):
        sm.qqplot(self.data[col], line='q')
        plt.title(f'QQ Plot {col}')
        plt.show()

    def plot_violin(self, col1, col2):
        print(self.data[col1].dtype)
        print(self.data[col2].dtype)
        # sns.violinplot(x=col1, y=col2, data=self.data)
        plt.title('Violin Plot of by Category')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()

    def hist_plot(self, col):
        sns.histplot(self.data[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

    def pair_plot(self, target_column):
        plot_data = df[[target_column] + [col for col in df.columns if col != target_column]]

        sns.pairplot(plot_data, kind='scatter')
        sns.pairplot(df, hue='Age', diag_kind='scatter')
        plt.tight_layout(pad=8.0)
        plt.show()

data = Data()
data.data_desc()
df = data.get_data()

# perform pre-processing and save the pre-pocessed file
preproc = PreProc(df)
data.save_data("preprocessed.csv")

eda = EDA(df)

eda.visual_missing_value()
eda.plot_corr()

feature = 'HeatingCosts'
eda.eda_report(feature)

# get Polynomial Features X, add interaction terms
# y is the target: Price
X, y = preproc.get_transformed_X_y()

fs_ob = Feature_Sel(df, X, y)

model_ob = LModel(fs_ob.X, y)
model = model_ob.fit()

# analyzing the effect of pool_quality feature
model_ob.without_pool_effect(model, df, fs_ob.X.columns)

# model_type can be set to Ordinary, Lasso, and Ridge
model_ob.cross_val(k=10, model_type='Ordinary')

# baseline always predicts mean of prices.
baseline = BaseLine(X, y)
baseline.cross_val()

# baseline2 is a linear model on raw features
X, y = preproc.get_raw_X_y()
baseline2_ob = LModel(X, y)
baseline2 = baseline2_ob.cross_val()

# figures of two features
col1 = 'Bedrooms'
col2 = 'Price'
eda.statistic_print(col1, col2)
# eda.plot_scatter(col1, col2)
eda.plot_barplot(col1, col2)
eda.plot_violin(col1, col2)
# eda.plot_cat_cat(col1, col2)

# plot qq-plot of feature: SquareFootageHouse
eda.plot_qqplot('SquareFootageHouse')
