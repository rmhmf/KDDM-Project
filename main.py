import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


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
        self.data_cleaning()
        self.data_transforming()

    def data_cleaning(self):
        self.remove_empty_rows()
        self.clean_location()

    def data_transforming(self):
        self.transform_hasphotovoltaics()
        self.transform_heatingtype()
        self.transform_hasfiberglass()
        self.transform_isfurnished()
        self.transform_datesinceforsale()
        self.transform_windowmodelnames()

    def remove_empty_rows(self):
        print(f"{self.data.isnull().all(axis=1).sum()} number of rows has been deleted.")
        self.data.dropna(how='all', inplace=True)

    def clean_location(self):
        location_misspell = {
            'Suburbann': 'Suburban'
        }
        self.data['Location'] = self.data['Location'].replace(location_misspell)

    def transform_hasphotovoltaics(self):
        self.data['HasPhotovoltaics'] = self.data['HasPhotovoltaics'].map({True: 1, False: 0}).astype(pd.Int32Dtype())

    def transform_heatingtype(self):
        unified_name = {
            'Oil Heating': 'Oil',
            'Electric': 'Electricity',
        }
        self.data['HeatingType'] = self.data['HeatingType'].replace(unified_name)

    def transform_hasfiberglass(self):
        self.data['HasFiberglass'] = self.data['HasFiberglass'].map({True: 1, False: 0}).astype(pd.Int32Dtype())

    def transform_isfurnished(self):
        self.data['IsFurnished'] = self.data['IsFurnished'].map({True: 1, False: 0}).astype(pd.Int32Dtype())

    def transform_datesinceforsale(self):
        self.data['DateSinceForSale'] = pd.to_datetime(self.data['DateSinceForSale'], format="%Y-%m-%d")
        mdate = self.data['DateSinceForSale'].max()

        def to_months(x):
            return (mdate.year - x.year) * 12 + (mdate.month - x.month)
        self.data['DateSinceForSale'] = self.data['DateSinceForSale'].apply(to_months)

    def transform_windowmodelnames(self):
        self.data.loc[self.data['WindowModelNames'].str.contains('Wood'), 'WindowModelNames'] = 'Wood'
        self.data.loc[self.data['WindowModelNames'].str.contains('Steel'), 'WindowModelNames'] = 'Steel'
        self.data.loc[self.data['WindowModelNames'].str.contains('Aluminum'), 'WindowModelNames'] = 'Aluminum'




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
            print(self.data[col].value_counts())
            sns.countplot(data=self.data, x=col)
        plt.title(f"Bar chart of column {col}", fontsize=18)
        plt.xticks(fontsize=24)
        plt.ylabel('Values')
        plt.tight_layout(pad=2.0)
        plt.show()


data = Data()
data.data_desc()
df = data.get_data()

preproc = PreProc(df)

eda = EDA(df)
col = 'Price'
eda.statistic_print(col)
eda.visual_missing_value(col)
eda.plot_boxplot(col)
eda.plot_count_bar(col)
