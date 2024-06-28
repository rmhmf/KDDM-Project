import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats


# The class that perform feature selection
class Feature_Sel:

    def __init__(self, data, X, y):
        self.data = data
        self.X = X
        self.sf = list(X.columns)
        self.y = y
        
        self.mutual_info_fs()
        self.vif()
        self.linear_fs()

    # selecting feature using mutual information as step 1
    def mutual_info_fs(self):
        threshold = .1
        mi_scores = mutual_info_regression(self.X, self.y)

        mi_scores = pd.Series(mi_scores, name="MI Scores", index=self.X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)

        # sns.histplot(mi_scores[mi_scores > threshold], kde=True)
        # plt.xlabel('Scores')
        # plt.ylabel('Frequency')
        # plt.title('Distribution of Scores')
        # plt.show()

        # print(f'Selected features:\n {mi_scores[mi_scores > threshold]}')
        self.sf = list((mi_scores[mi_scores > threshold]).index)
        print(f'Feature size after mutual information: {len(self.sf)}')
        self.X = self.X[self.sf]

    # for the next step use VIF for removing the highly correlated features
    # as an alternative -the commented code- we could use correlation matrix
    def vif(self):
        # corr_matrix = self.X.corr().abs()

        # threshold = 0.85
        # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        # for drop_e in to_drop:
        #     self.sf.remove(drop_e)
        #     print(f'{drop_e} is removed')
        # self.X = self.X[self.sf]
        # print(len(self.sf))

        def calculate_vif(X):
            vif_data = pd.DataFrame()
            vif_data["feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            return vif_data

        threshold = 15
        while True:
            vif_data = calculate_vif(self.X)
            max_vif = vif_data["VIF"].max()
            if max_vif > threshold:
                drop_feature = vif_data.sort_values("VIF", ascending=False).iloc[0]["feature"]
                self.X = self.X.drop(columns=[drop_feature])
                self.sf.remove(drop_feature)
                print(f"Dropped {drop_feature} with VIF {max_vif}")
            else:
                break

        print(f'Feature size after vif: {len(self.sf)}')

    # For the final step, I fitted a linear model and removed insignificant features
    def linear_fs(self, signif_lev=.05):

        # X = sm.add_constant(self.X[self.sf])
        # model = sm.OLS(self.y, X).fit()
        # print(model.summary())
            
        while True:
            X = sm.add_constant(self.X[self.sf])
            model = sm.OLS(self.y, X).fit()
            pvalues = model.pvalues
            
            max_pvalue = pvalues.max()

            if max_pvalue > signif_lev:
                excluded_feature = pvalues.idxmax()
                self.sf.remove(excluded_feature)
                print(f' Feature {excluded_feature} is removed.')
            else: break

        
        self.X = self.X[self.sf]
        print(f'Feature size after linear fs: {len(self.sf)}')
        

# The Linear model class
class LModel:

    def __init__(self, X, y):
        self.X = X
        self.y = y 

    def fit(self):
        X = sm.add_constant(self.X)
        model = sm.OLS(self.y, X).fit()
        print(model.summary())
        return model

    # prints MSE, MAE, R-Squared and returns predicted values
    def evaluate(self, model, X, y):
        X = sm.add_constant(X)
        y_pred = model.predict(X)

        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        print("Mean Squared Error (MSE):", mse)
        print("Mean Absolute Error (MAE):", mae)

        print(f"R-squared: {model.rsquared}")

        return y_pred

    # fits a model given the train data, returns r-squared and predicted values
    # model type can be set to Ordinary, Lasso, Ridge
    def fit_and_predict(self, X_train, y_train, X_test, model_type):
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)
        if model_type == 'Ordinary':
            model = sm.OLS(y_train, X_train).fit()
        elif model_type == 'Lasso':
            model = sm.OLS(y_train, X_train)
            model = model.fit_regularized(method='elastic_net', alpha=0.1, L1_wt=1.0)
        elif model_type == 'Ridge':
            model = sm.OLS(y_train, X_train)
            model = model.fit_regularized(method='elastic_net', alpha=0.1, L1_wt=0.0)

        y_train_pred = model.predict(X_train)
        rsquared = r2_score(y_train, y_train_pred)

        predictions = model.predict(X_test)
        return predictions, rsquared

    # cross validation evaluation on linear model
    def cross_val(self, k=10, model_type='Ordinary'):
        kf = KFold(n_splits=k, shuffle=True, random_state=1)

        Rsquared = []
        MSE = []
        MAE = []
        # Loop through each fold
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            y_pred, rsq = self.fit_and_predict(X_train, y_train, X_test, model_type)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            Rsquared.append(rsq)
            MSE.append(mse)
            MAE.append(mae)

        print('R-Squared:')
        print(Rsquared)
        print(f'mean: {np.mean(Rsquared)}, std: {np.std(Rsquared)}')
        print('MSE:')
        print(MSE)
        print(f'mean: {np.mean(MSE)}, std: {np.std(MSE)}')
        print(' MAE')
        print(MAE)
        print(f'mean: {np.mean(MAE)}, std: {np.std(MAE)}')

    # Analyze the effect of PoolQuality feature
    def without_pool_effect(self, model, df, sf, only_inter=True):
        dfc = df.copy()
        dfc['SquareFootageGarden'] = 6
        target = 'Price'

        all_cols = list(dfc.columns)
        all_cols.remove(target)
        respons = all_cols

        X = dfc[respons]
        y = dfc[target]
        y.reset_index(drop=True, inplace=True)

        pf = PolynomialFeatures(degree=2, interaction_only=only_inter, include_bias=True)
        X = pf.fit_transform(X)

        pf_feature_names = pf.get_feature_names_out(input_features=respons)
        X = pd.DataFrame(X, columns=list(pf_feature_names))
        X = X[sf]

        pred = self.evaluate(model, self.X, self.y)
        pred_without_pool = self.evaluate(model, X, y)
        price_changes = pred - pred_without_pool
        print((price_changes).describe())
        # price_changes.to_csv('a.csv')
        t_statistic, p_value = stats.ttest_rel(pred, pred_without_pool)
        print(t_statistic)
        print(p_value)

        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 5))
        sns.kdeplot(price_changes, fill=True, color="b", label="Density")
        plt.axvline(x=0, linestyle="--", color="r", label="No Difference")

        plt.title("Density Function of Differences in Predicted Values")
        plt.xlabel("Difference")
        plt.ylabel("Density")
        plt.legend()
        plt.show()


# a model that always predicts mean of prices
class BaseLine:

    def __init__(self, X, y):
        self.X = X 
        self.y = y 

    def cross_val(self, k=10):
        kf = KFold(n_splits=k, shuffle=True, random_state=1)

        MSE = []
        MAE = []
        for train_index, test_index in kf.split(self.X):

            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            
            y_pred = pd.Series([np.mean(y_train)] * len(y_test), index=y_test.index)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            MSE.append(mse)
            MAE.append(mae)

        print('MSE:')
        print(MSE)
        print(f'mean: {np.mean(MSE)}, std: {np.std(MSE)}')
        print(' MAE')
        print(MAE)
        print(f'mean: {np.mean(MAE)}, std: {np.std(MAE)}')
