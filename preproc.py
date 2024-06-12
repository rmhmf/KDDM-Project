import pandas as pd


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
        self.clean_location()
        self.clean_heatingtype()
        self.clean_windowmodelnames()

    def remove_empty_rows(self):
        print(f"{self.data.isnull().all(axis=1).sum()} number of rows has been deleted.")
        self.data.dropna(how='all', inplace=True)

    def clean_location(self):
        location_misspell = {
            'Suburbann': 'Suburban'
        }
        self.data['Location'] = self.data['Location'].replace(location_misspell)

    def clean_heatingtype(self):
        unified_name = {
            'Oil Heating': 'Oil',
            'Electric': 'Electricity',
        }
        self.data['HeatingType'] = self.data['HeatingType'].replace(unified_name)

    def clean_windowmodelnames(self):
        self.data.loc[self.data['WindowModelNames'].str.contains('Wood'), 'WindowModelNames'] = 'Wood'
        self.data.loc[self.data['WindowModelNames'].str.contains('Steel'), 'WindowModelNames'] = 'Steel'
        self.data.loc[self.data['WindowModelNames'].str.contains('Aluminum'), 'WindowModelNames'] = 'Aluminum'


class Outlier:
    def __init__(self, data):
        self.data = data


class MissingV:
    def __init__(self, data):
        self.data = data


class Transform:
    def __init__(self, data):
        self.data = data

        self.transform_bedrooms()
        self.transform_bathrooms()
        self.transform_poolquality()
        self.transform_hasphotovoltaics()
        self.transform_hasfiberglass()
        self.transform_isfurnished()
        self.transform_datesinceforsale()
        self.transform_hasfireplace()
        self.transform_kitchensquality()
        self.transform_bathroomsquality()
        self.transform_bedroomsquality()
        self.transform_livingroomsquality()

    def transform_bedrooms(self):
        self.data['Bedrooms'] = self.data['Bedrooms'].astype(pd.Int32Dtype())

    def transform_bathrooms(self):
        self.data['Bathrooms'] = self.data['Bathrooms'].astype(pd.Int32Dtype())

    def transform_poolquality(self):
        self.quality_to_quantity('PoolQuality')

    def transform_hasphotovoltaics(self):
        self.data['HasPhotovoltaics'] = self.data['HasPhotovoltaics'].map({True: 1, False: 0}).astype(pd.Int32Dtype())


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

    def transform_hasfireplace(self):
        self.data['HasFireplace'] = self.data['HasFireplace'].map({True: 1, False: 0}).astype(pd.Int32Dtype())
    
    def transform_kitchensquality(self):
        self.quality_to_quantity('KitchensQuality')

    def transform_bathroomsquality(self):
        self.quality_to_quantity('BathroomsQuality')

    def transform_bedroomsquality(self):
        self.quality_to_quantity('BedroomsQuality')

    def transform_livingroomsquality(self):
        self.quality_to_quantity('LivingRoomsQuality')

    def quality_to_quantity(self, col):
        map_dic = {
            'Poor': 1,
            'Good': 2,
            'Excellent': 3,
        }
        self.data[col] = self.data[col].map(map_dic).astype(pd.Int32Dtype())
