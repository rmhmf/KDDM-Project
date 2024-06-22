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


class Feature_Sel:

    def __init__(self, data, X, y):
        self.data = data
        self.X = X
        self.sf = list(X.columns)
        self.y = y
        
        self.mutual_info_fs()
        self.vif()
        self.linear_fs()

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
        

class LModel:

    def __init__(self, X, y):
        self.X = X
        self.y = y 

    def fit(self):
        X = sm.add_constant(self.X)
        model = sm.OLS(self.y, X).fit()
        print(model.summary())
        return model

    def evaluate(self, model, X, y):
        X = sm.add_constant(X)
        y_pred = model.predict(X)

        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        print("Mean Squared Error (MSE):", mse)
        print("Mean Absolute Error (MAE):", mae)

        print(f"R-squared: {model.rsquared}")

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

    def cross_val(self, k=10, model_type='Ordinary'):
        kf = KFold(n_splits=k, shuffle=True, random_state=1)

        Rsquared = []
        MSE = []
        MAE = []
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            y_pred, rsq = self.fit_and_predict(X_train, y_train, X_test, model_type)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            Rsquared.append(rsq)
            MSE.append(mse)
            MAE.append(mae)

        print(Rsquared)
        print(np.mean(Rsquared))
        print(MSE)
        print(np.mean(MSE))
        print(MAE)
        print(np.mean(MAE))


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

        print(MSE)
        print(np.mean(MSE))
        print(MAE)
        print(np.mean(MAE))
