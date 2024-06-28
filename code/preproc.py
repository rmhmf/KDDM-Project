import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2


# class for pre-processing
class PreProc:
    def __init__(self, data):
        self.data = data

        self.clean = Clean(self.data)
        self.outlier = Outlier(self.data)
        self.missing_v = MissingV(self.data)
        self.transform = Transform(self.data)

        self.missing_v.heatingcosts()
        self.transform.heatingcosts()

        self.missing_v.impute_remained()

        # self.outlier.pca_outlier()
        # self.outlier.mahalanobis_outlier()
    
    # add binomial terms to the features
    def get_transformed_X_y(self, only_interactions=True):
        return self.transform.add_features(only_interactions)
    
    # returns features X as predictors and price as y, is response
    def get_raw_X_y(self):
        return self.transform.get_features()


# class for performing data cleaning
class Clean:
    def __init__(self, data):
        self.data = data

        self.remove_empty_rows()
        self.location()
        self.heatingtype()
        self.windowmodelnames()

    # removing empty rows
    def remove_empty_rows(self):
        print(f"{self.data.isnull().all(axis=1).sum()} number of rows has been deleted.")
        self.data.dropna(how='all', inplace=True)

    # Fix misspelling in location
    def location(self):
        location_misspell = {
            'Suburbann': 'Suburban'
        }
        self.data['Location'] = self.data['Location'].replace(location_misspell)

    # make the values of HeatingType homogenous
    def heatingtype(self):
        unified_name = {
            'Oil Heating': 'Oil',
            'Electric': 'Electricity',
        }
        self.data['HeatingType'] = self.data['HeatingType'].replace(unified_name)

    # reduce the values to the type of window: Wood, Steel, Aluminum
    def windowmodelnames(self):
        self.data.loc[self.data['WindowModelNames'].str.contains('Wood'), 'WindowModelNames'] = 'Wood'
        self.data.loc[self.data['WindowModelNames'].str.contains('Steel'), 'WindowModelNames'] = 'Steel'
        self.data.loc[self.data['WindowModelNames'].str.contains('Aluminum'), 'WindowModelNames'] = 'Aluminum'


# Class to address outliers
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

    # Use PCA to identify multivariate outliers
    def pca_outlier(self):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.data)

        print(self.data.shape)
        pca = PCA(n_components=2)
        data_transformed = pca.fit_transform(scaled_features)
        pca_df = pd.DataFrame(data_transformed, columns=['PCA-1', 'PCA-2'])

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PCA-1', y='PCA-2', data=pca_df, alpha=0.5)
        plt.title('PCA Visualization of the Housing Data')
        plt.xlabel('PCA Dimension 1')
        plt.ylabel('PCA Dimension 2')
        plt.show()

        print(pca.explained_variance_ratio_)
        print(np.sum(pca.explained_variance_ratio_))

    # Use mahalanobis distance to detect outliers
    def mahalanobis_outlier(self):

        covariance_matrix = np.cov(self.data.astype(float).values.T)
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        mean = np.mean(self.data.astype(float).values, axis=0)

        def mahalanobis_distance(x):
            x_minus_mean = x.astype(float) - mean
            md = np.sqrt(x_minus_mean @ inv_covariance_matrix @ x_minus_mean.T)
            return md

        results = self.data.apply(mahalanobis_distance, axis=1)
        results_df = pd.DataFrame(results, columns=['MD'])
        print(results_df.describe())

        dof = len(self.data.columns)
        threshold = chi2.ppf(0.99, df=dof)
        print(threshold)


# class for missing value imputation
class MissingV:
    def __init__(self, data):
        self.data = data
        self.bedrooms()
        self.bathrooms()
        self.poolquality()
        self.hasphotovoltaics()
        self.previousownername()

    # Use base rule to impute missing values using SquareFootageHouse
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

    # Use base rule to impute missing values using SquareFootageHouse
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

    # Use SquareFootageGarden to impute missing values of PoolQuality
    def poolquality(self):
        def fill_mar(row):
            if row['SquareFootageGarden'] == 6:
                return 'NoPool'
            elif row['SquareFootageGarden'] == 14:
                return 'Poor'
            elif row['SquareFootageGarden'] == 16:
                return 'Good'
            else:
                return 'Excellent'

        self.data['PoolQuality'] = self.data.apply(fill_mar, axis=1)

    # fit a linear model for imputing missing values of HeatingCosts
    # using three predictors 'SquareFootageHouse', 'HeatingType_Oil', 'Price'
    def heatingcosts(self):
        respons = ['SquareFootageHouse', 'HeatingType_Oil', 'Price']
        lm = LinearM(self.data, 'OLS', "HeatingCosts", respons)
        model, selected_features = lm.feature_selection()

        # model2 = sm.OLS(y, phi).fit()

        x_pred_d = self.data[respons].dropna()

        pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)
        x_pred = pf.fit_transform(x_pred_d)

        pf_feature_names = pf.get_feature_names_out(input_features=respons)
        x_pred = pd.DataFrame(x_pred, columns=list(pf_feature_names))[selected_features]

        predictions = model.predict(x_pred).to_frame()

        predictions.set_index(x_pred_d.index, inplace=True)

        def impute(row):
            if pd.isna(row['HeatingCosts']):
                if row.name in predictions.index:
                    return predictions.loc[row.name]
            else:
                return row['HeatingCosts']                
            
        self.data['HeatingCosts'] = self.data.apply(impute, axis=1)

    # drop the column HasPhotovoltaics
    def hasphotovoltaics(self):
        self.data.drop('HasPhotovoltaics', axis=1, inplace=True)

    # drop the feature PreviousOwnerName
    def previousownername(self):
        self.data.drop('PreviousOwnerName', axis=1, inplace=True)

    # Use KNN to address the remaining missing values
    def impute_remained(self):
        col_rep = ['HouseColor_Gray', 'HouseColor_Green', 'HouseColor_White', 'HouseColor_Yellow']
        self.knn_impute(col_rep, 1, int)

        self.knn_impute(['Age'], 3, int)

        col_rep = ['Bedrooms', 'Bathrooms', 'SquareFootageHouse', 'Location_Rural', 'Location_Suburban', 'Location_Urban']
        self.knn_impute(col_rep, 1, int)

        col_rep = ['HeatingCosts', 'Price']
        self.knn_impute(col_rep, 3, float)

    # Knn to impute missing variables of col_rep features
    def knn_impute(self, col_rep, n, type):
        imputer = KNNImputer(n_neighbors=n)
        
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(self.data)

        imputed_df_norm = imputer.fit_transform(normalized_data)
        imputed_df = scaler.inverse_transform(imputed_df_norm).astype(type)
        imputed_df = pd.DataFrame(imputed_df, columns=self.data.columns, index=self.data.index)

        miss_ind = self.data[col_rep].isnull().any(axis=1)
        self.data.loc[miss_ind, col_rep] = imputed_df.loc[miss_ind, col_rep]
        

# class for performing feature transformation
class Transform:
    def __init__(self, data):
        self.data = data

        self.bedrooms()
        self.bathrooms()
        self.squarefootagehouse()
        self.location()
        self.age()
        self.poolquality()
        # self.hasphotovoltaics()
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
        # self.heatingcosts()
        self.windowmodelnames()
        self.price()

    # transform bedrooms to Integer
    def bedrooms(self):
        self.data['Bedrooms'] = self.data['Bedrooms'].astype(pd.Int32Dtype())

    # transform bathroom to Integer
    def bathrooms(self):
        self.data['Bathrooms'] = self.data['Bathrooms'].astype(pd.Int32Dtype())

    # some feature transformation tested, at the end I left the feature as was
    def squarefootagehouse(self):
        # self.data['SquareFootageHouse'] = self.data['SquareFootageHouse'].apply(np.sqrt)
        # transformed_data, lambda_value = stats.boxcox(self.data['SquareFootageHouse'])
        # self.data['SquareFootageHouse'] = transformed_data
        pass

    # transformed location as a categorical data to one-hot encoding
    def location(self):
        self.to_one_hot('Location')
        self.data.drop('Location', axis=1, inplace=True)

    # many transformations tested, at the end I left the feature as was
    def age(self):

        def binary(x):
            if not pd.isna(x):
                if x>40:
                    return 1
                else: return 0
            else:
                return 1

        # ind = self.data.columns.get_loc('Age')
        # new_col = self.data['Age'].apply(binary).astype(pd.Int32Dtype())

        # self.data.insert(ind+1,'Age_binary', new_col)

        # def bins(x):
        #     if not pd.isna(x):
        #         return x // 10
        #     else:
        #         return np.nan
        
        # self.data['Age'] = self.data['Age'].apply(bins).astype(pd.Int32Dtype())

        # self.add_bool_col('Age')

        max_age = self.data['Age'].max()
        # self.data['Age'] = self.data['Age'].apply(self.reflect_sqrt, max=max_age)

    # ordinal encoding
    def poolquality(self):
        self.quality_to_quantity('PoolQuality')
        
    # transform bool to 0-1
    def hasphotovoltaics(self):
        self.data['HasPhotovoltaics'] = self.data['HasPhotovoltaics'].map({True: 1, False: 0}).astype(pd.Int32Dtype())
        # self.add_bool_col('HasPhotovoltaics')

    # transformed HeatingType as a categorical data to one-hot encoding
    def heatingtype(self):
        self.to_one_hot('HeatingType')
        self.data.drop('HeatingType', axis=1, inplace=True)

    # transform bool to 0-1
    def hasfiberglass(self):
        self.data['HasFiberglass'] = self.data['HasFiberglass'].map({True: 1, False: 0}).astype(pd.Int32Dtype())

    # transform bool to 0-1
    def isfurnished(self):
        self.data['IsFurnished'] = self.data['IsFurnished'].map({True: 1, False: 0}).astype(pd.Int32Dtype())

    # transform the date to the number of month that has passed since the last date
    def datesinceforsale(self):
        self.data['DateSinceForSale'] = pd.to_datetime(self.data['DateSinceForSale'], format="%Y-%m-%d")
        mdate = self.data['DateSinceForSale'].max()

        def to_months(x):
            return (mdate.year - x.year) * 12 + (mdate.month - x.month)
            # return mdate.year - x.year
        self.data['DateSinceForSale'] = self.data['DateSinceForSale'].apply(to_months).astype(pd.Int32Dtype())
        # self.data['DateSinceForSale'] = self.data['DateSinceForSale'].apply(self.log, c=2)

    # transformed HouseColor as a categorical data to one-hot encoding
    def housecolor(self):
        self.to_one_hot('HouseColor')
        self.data.drop('HouseColor', axis=1, inplace=True)

    # at the end I dropped the feature
    def previousownername(self):
        # self.to_one_hot('PreviousOwnerName')
        self.data.drop('PreviousOwnerName', axis=1, inplace=True)

    # transform bool to 0-1
    def hasfireplace(self):
        self.data['HasFireplace'] = self.data['HasFireplace'].map({True: 1, False: 0}).astype(pd.Int32Dtype())
    
    # ordinal encoding
    def kitchensquality(self):
        self.quality_to_quantity('KitchensQuality')

    # ordinal encoding
    def bathroomsquality(self):
        self.quality_to_quantity('BathroomsQuality')

    # ordinal encoding
    def bedroomsquality(self):
        self.quality_to_quantity('BedroomsQuality')

    # ordinal encoding
    def livingroomsquality(self):
        self.quality_to_quantity('LivingRoomsQuality')

    # transform it to Int type
    def squarefootagegarden(self):
        self.data['SquareFootageGarden'] = self.data['SquareFootageGarden'].astype(pd.Int32Dtype())

    # at the end I left the feature as was
    # It was a bit right skewed, so I thought maybe transforming by log get better results, which didn't
    def previousownerrating(self):
        # self.data['PreviousOwnerRating'] = self.data['PreviousOwnerRating'].apply(np.log)
        pass

    # at the end I left the feature as was
    # It was a bit right skewed, so I thought maybe transforming by sqrt get better results, which didn't
    def heatingcosts(self):
        # self.data['HeatingCosts'] = self.data['HeatingCosts'].apply(np.sqrt)
        pass

    # transformed WindowModelNames as a categorical data to one-hot encoding
    def windowmodelnames(self):
        self.to_one_hot('WindowModelNames')
        self.data.drop('WindowModelNames', axis=1, inplace=True)

    # I tried different feature transformation, at the end I left it as was
    def price(self):
        # self.data['Price'] = self.data['Price'].apply(np.log)

        # transformed_data, lambda_value = stats.boxcox(self.data['Price'])
        # self.data['Price'] = transformed_data
        pass

    # Transform quality feature to numeric
    def quality_to_quantity(self, col):
        map_dic = {
            'NoPool': 0,
            'Poor': 1,
            'Good': 2,
            'Excellent': 3,
        }
        self.data[col] = self.data[col].map(map_dic).astype(pd.Int32Dtype())

    # transform categorical feature to one-hot encoding
    def to_one_hot(self, col):
        loc_ind = self.data.columns.get_loc(col)
        one_hot_cols = pd.get_dummies(self.data[col], prefix=col, dtype=int)

        nan_ind = self.data[col].isna()
        one_hot_cols.iloc[nan_ind, :] = np.nan 

        # self.data.drop(col, axis=1, inplace=True)

        for i, col in enumerate(one_hot_cols.columns):
            self.data.insert(loc_ind + i, col, one_hot_cols[col])

    # transformation for left skewed data
    def reflect_sqrt(self, x, max):
        return np.sqrt(max + 1 - x)

    # add a boolean feature that is 1 whenever col is missed
    # I used this feature to see if the missing values are dependent on something
    def add_bool_col(self, col):
        ind = self.data.columns.get_loc(col)
        def bool_f(x):
            if pd.isna(x):
                return 0
            else:
                return 1
        
        new_col = self.data[col].apply(bool_f).astype(pd.Int32Dtype())
        self.data.insert(ind+1, col+'_ismissed', new_col)

    # adds binomial terms to feature and returns it as X, also return response y -Price-
    def add_features(self, only_inter):
        target = 'Price'

        all_cols = list(self.data.columns)
        all_cols.remove(target)
        respons = all_cols

        X = self.data[respons]
        y = self.data[target]
        y.reset_index(drop=True, inplace=True)

        pf = PolynomialFeatures(degree=2, interaction_only=only_inter, include_bias=True)
        X = pf.fit_transform(X)

        pf_feature_names = pf.get_feature_names_out(input_features=respons)
        X = pd.DataFrame(X, columns=list(pf_feature_names))

        return X, y
    
    # returns raw features X and response y
    def get_features(self):
        target = 'Price'

        all_cols = list(self.data.columns)
        all_cols.remove(target)
        respons = all_cols

        X = self.data[respons].astype(float)
        X.reset_index(drop=True, inplace=True)

        y = self.data[target].astype(float)
        y.reset_index(drop=True, inplace=True)

        return X, y

    def log(self, x, c):
        return np.log(x + c)


# Linear model for missing value imputation
class LinearM:

    def __init__(self, data, method, target, respons=None):
        self.data = data
        self.method = method
        if not respons:
            numerical_cols = self.data.select_dtypes(include=['float64', 'Int32', 'Float64', 'Int64']).columns
            cols_list = list(numerical_cols)
            cols_list.remove(target)
            self.respons = cols_list
        else:
            self.respons = respons
        self.target = target
    
    # get coulumns that have no missing value
    def columns_without_nan(self, cols):
        columns_without_nan = self.data.columns[self.data.notna().all()].tolist()
        return columns_without_nan

    # drop rows with missing values
    def drop_nan(self):
        valued_df = self.data[self.respons+[self.target]].dropna().astype(float)
        valued_df.reset_index(inplace=True)
        return valued_df

    # adds interaction terms of features
    # performs feature selection that does not violate ANOVA
    # return the final model
    def feature_selection(self, signif_lev=.05):

        df = self.drop_nan()

        X = df[self.respons]
        y = df[[self.target]]

        pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)
        X = pf.fit_transform(X)

        pf_feature_names = pf.get_feature_names_out(input_features=self.respons)
        X = pd.DataFrame(X, columns=list(pf_feature_names))
        all_features = list(pf_feature_names)

        sole_terms = self.respons.copy()
        interaction_terms = [item for item in all_features if item not in sole_terms]

        feasible_terms = interaction_terms.copy()
        best_features = all_features.copy()

        # remove insignificat features one by one
        while len(best_features) > 1:
            
            phi = X[best_features]
            model = sm.OLS(y, phi).fit()
            pvalues = model.pvalues

            feas_ind = []
            for ind, element in enumerate(best_features):
                if element in feasible_terms:
                    feas_ind.append(ind)
            
            max_pvalue = pvalues[feas_ind].max()

            if max_pvalue > signif_lev:
                excluded_feature = pvalues[feas_ind].idxmax()
                
                best_features.remove(excluded_feature)
                feasible_terms.remove(excluded_feature)

                parts = excluded_feature.split()
                for part in parts:
                    if part not in feasible_terms:
                        feasible_terms.append(part)
            else:
                break

        return model, best_features
