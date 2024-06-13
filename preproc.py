import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer


class PreProc:
    def __init__(self, data):
        self.data = data

        clean = Clean(self.data)
        outlier = Outlier(self.data)
        missing_v = MissingV(self.data)
        transform = Transform(self.data)


class Clean:
    def __init__(self, data):
        self.data = data

        self.remove_empty_rows()
        self.location()
        self.heatingtype()
        self.windowmodelnames()

    def remove_empty_rows(self):
        print(f"{self.data.isnull().all(axis=1).sum()} number of rows has been deleted.")
        self.data.dropna(how='all', inplace=True)

    def location(self):
        location_misspell = {
            'Suburbann': 'Suburban'
        }
        self.data['Location'] = self.data['Location'].replace(location_misspell)

    def heatingtype(self):
        unified_name = {
            'Oil Heating': 'Oil',
            'Electric': 'Electricity',
        }
        self.data['HeatingType'] = self.data['HeatingType'].replace(unified_name)

    def windowmodelnames(self):
        self.data.loc[self.data['WindowModelNames'].str.contains('Wood'), 'WindowModelNames'] = 'Wood'
        self.data.loc[self.data['WindowModelNames'].str.contains('Steel'), 'WindowModelNames'] = 'Steel'
        self.data.loc[self.data['WindowModelNames'].str.contains('Aluminum'), 'WindowModelNames'] = 'Aluminum'


class Outlier:
    def __init__(self, data):
        self.data = data
        self.squarefootagehouse()
        self.age()
        self.heatingcosts()
        self.price()
    
    def squarefootagehouse(self):
        threshold = 200
        self.data.loc[self.data['SquareFootageHouse'] > threshold, 'SquareFootageHouse'] = np.nan
        self.data.loc[self.data['SquareFootageHouse'] < 0, 'SquareFootageHouse'] = np.nan

    def heatingcosts(self):
            threshold = 300
            self.data.loc[self.data['HeatingCosts'] > threshold, 'HeatingCosts'] = np.nan
            self.data.loc[self.data['HeatingCosts'] < 0, 'HeatingCosts'] = np.nan

    def age(self):
        self.data.loc[self.data['Age'] < 0, 'Age'] = np.nan

    def price(self):
        threshold = 1000
        self.data.loc[self.data['Price'] > threshold, 'Price'] = np.nan


class MissingV:
    def __init__(self, data):
        self.data = data
        self.bedrooms()
        self.bathrooms()

    def bedrooms(self):
        # Using the correlation between Bedrooms and SquareFootageHouse
        stat = self.data.groupby('Bedrooms')['SquareFootageHouse'].agg(['mean', 'std']).to_numpy()
        priors = self.data['Bedrooms'].value_counts(normalize=True).to_numpy()

        def predic_label(row):
            if pd.isna(row['Bedrooms']) & pd.isna(row['SquareFootageHouse']):
                return np.nan
            elif pd.isna(row['Bedrooms']):
                x = row['SquareFootageHouse']
                likelihoods = [stats.norm.pdf(x, loc=entry[0], scale=entry[1]) for entry in stat]
                posters = likelihoods * priors
                return np.argmax(posters) + 1
            else:
                return row['Bedrooms']
        
        self.data['Bedrooms'] = self.data.apply(predic_label, axis=1)

    def bathrooms(self):
        # Using the correlation between Bathrooms and SquareFootageHouse
        stat = self.data.groupby('Bathrooms')['SquareFootageHouse'].agg(['mean', 'std']).to_numpy()
        priors = self.data['Bathrooms'].value_counts(normalize=True).to_numpy()

        def predic_label(row):
            if pd.isna(row['Bathrooms']) & pd.isna(row['SquareFootageHouse']):
                return np.nan
            elif pd.isna(row['Bathrooms']):
                x = row['SquareFootageHouse']
                likelihoods = [stats.norm.pdf(x, loc=entry[0], scale=entry[1]) for entry in stat]
                posters = likelihoods * priors
                return np.argmax(posters) + 1
            else:
                return row['Bathrooms']
        
        self.data['Bathrooms'] = self.data.apply(predic_label, axis=1)
class Transform:
    def __init__(self, data):
        self.data = data

        self.bedrooms()
        self.bathrooms()
        self.location()
        # self.age()
        self.poolquality()
        self.hasphotovoltaics()
        self.heatingtype()
        self.hasfiberglass()
        self.isfurnished()
        self.datesinceforsale()
        self.housecolor()
        # self.previousownername()
        self.hasfireplace()
        self.kitchensquality()
        self.bathroomsquality()
        self.bedroomsquality()
        self.livingroomsquality()
        self.squarefootagegarden()
        self.previousownerrating()
        self.heatingcosts()
        self.windowmodelnames()
        self.price()

    def bedrooms(self):
        self.data['Bedrooms'] = self.data['Bedrooms'].astype(pd.Int32Dtype())

    def bathrooms(self):
        self.data['Bathrooms'] = self.data['Bathrooms'].astype(pd.Int32Dtype())

    def location(self):
        self.to_one_hot('Location')

    def age(self):
        max_age = self.data['Age'].max()
        self.data['Age'] = self.data['Age'].apply(self.reflect_sqrt, max=max_age)

    def poolquality(self):
        self.quality_to_quantity('PoolQuality')

    def hasphotovoltaics(self):
        self.data['HasPhotovoltaics'] = self.data['HasPhotovoltaics'].map({True: 1, False: 0}).astype(pd.Int32Dtype())

    def heatingtype(self):
        self.to_one_hot('HeatingType')

    def hasfiberglass(self):
        self.data['HasFiberglass'] = self.data['HasFiberglass'].map({True: 1, False: 0}).astype(pd.Int32Dtype())

    def isfurnished(self):
        self.data['IsFurnished'] = self.data['IsFurnished'].map({True: 1, False: 0}).astype(pd.Int32Dtype())

    def datesinceforsale(self):
        self.data['DateSinceForSale'] = pd.to_datetime(self.data['DateSinceForSale'], format="%Y-%m-%d")
        mdate = self.data['DateSinceForSale'].max()

        def to_months(x):
            return (mdate.year - x.year) * 12 + (mdate.month - x.month)
        self.data['DateSinceForSale'] = self.data['DateSinceForSale'].apply(to_months)

    def housecolor(self):
        self.to_one_hot('HouseColor')

    def previousownername(self):
        self.to_one_hot('PreviousOwnerName')

    def hasfireplace(self):
        self.data['HasFireplace'] = self.data['HasFireplace'].map({True: 1, False: 0}).astype(pd.Int32Dtype())
    
    def kitchensquality(self):
        self.quality_to_quantity('KitchensQuality')

    def bathroomsquality(self):
        self.quality_to_quantity('BathroomsQuality')

    def bedroomsquality(self):
        self.quality_to_quantity('BedroomsQuality')

    def livingroomsquality(self):
        self.quality_to_quantity('LivingRoomsQuality')

    def squarefootagegarden(self):
        self.data['SquareFootageGarden'] = self.data['SquareFootageGarden'].astype(pd.Int32Dtype())

    def previousownerrating(self):
        self.data['PreviousOwnerRating'] = self.data['PreviousOwnerRating'].apply(np.log)

    def heatingcosts(self):
        self.data['HeatingCosts'] = self.data['HeatingCosts'].apply(np.sqrt)

    def windowmodelnames(self):
        self.to_one_hot('WindowModelNames')

    def price(self):
        self.data['Price'] = self.data['Price'].apply(np.log)
        # transformed_data, lambda_value = stats.boxcox(self.data['Price'])
        # self.data['Price'] = transformed_data

    def quality_to_quantity(self, col):
        map_dic = {
            'Poor': 1,
            'Good': 2,
            'Excellent': 3,
        }
        self.data[col] = self.data[col].map(map_dic).astype(pd.Int32Dtype())

    def to_one_hot(self, col):
        loc_ind = self.data.columns.get_loc(col)
        one_hot_cols = pd.get_dummies(self.data[col], prefix=col, dtype=int)

        nan_ind = self.data[col].isna()
        one_hot_cols.iloc[nan_ind, :] = np.nan 

        # self.data.drop(col, axis=1, inplace=True)

        for i, col in enumerate(one_hot_cols.columns):
            self.data.insert(loc_ind + i, col, one_hot_cols[col])

    def reflect_sqrt(self, x, max):
        return np.sqrt(max + 1 - x)
            