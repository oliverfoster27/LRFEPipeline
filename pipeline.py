import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Imputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from itertools import combinations, chain
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


class LRFEPipeline(BaseEstimator, TransformerMixin):

    # Class Constructor
    def __init__(self, cv=3, inversion=False):
        self.cv = cv
        self.inversion = inversion
        self.fit_state = []

    @staticmethod
    def powerset(i):
        s = list(i)  # allows duplicate elements
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    def fit(self, X, y=None):

        self.fit_state = []
        model = Lasso
        for feat in X.columns:

            min_err = 10e6
            X_trial = pd.DataFrame()
            for deg in [0.5, 1, 2, 3, 4]:
                X_trial[deg] = X[feat] ** deg

            best_inv_state = 0
            for inv_state, target in enumerate([y, 1 / y]):

                if inv_state != 0 and self.inversion == False:
                    break

                for i, combo in enumerate(self.powerset(X_trial.columns), 1):

                    combo = list(combo)

                    kf = KFold(n_splits=self.cv, shuffle=True)
                    err = []
                    for train_index, test_index in kf.split(X_trial):
                        X_train, X_test = X_trial.loc[train_index, combo], X_trial.loc[test_index, combo]
                        y_train, y_test = target[train_index], target[test_index]
                        md = model().fit(X_train, y_train)
                        y_pred = md.predict(X_test)
                        err.append(mean_squared_error(y_test, y_pred))

                    avg_err = sum(err) / len(err)
                    if avg_err < min_err:
                        min_err = avg_err
                        best_combo, best_inv_state = combo, inv_state

            if best_inv_state:
                target = 1 / y
            else:
                target = y

            md = model().fit(X_trial[combo], target)
            operations = [(transf, coef) for transf, coef in zip(combo, md.coef_) if coef != 0]
            intercept = md.intercept_

            if best_inv_state:
                feature_str = ' + '.join(
                    ['{} * {} ^ {}'.format(coef, feat, int(transf)) for transf, coef in zip(combo, md.coef_) if coef != 0])
                feature_str = '{} + {}'.format(feature_str, intercept)
                feature_str = '1 / ({})'.format(feature_str)
            else:
                feature_str = feat

            d = {
                'feature': feature_str,
                'base_feature': feat,
                'operations': operations,
                'intercept': intercept,
                'inv_state': best_inv_state
            }
            self.fit_state.append(d)

        return self

    def transform(self, X, y=None):

        X_trans = X.copy()
        for state in self.fit_state:

            if state['inv_state'] == 1:
                X_trans[state['feature']] = 0.0
                for transf, coef in state['operations']:
                    if coef != 0:
                        X_trans[state['feature']] += coef * (X[state['base_feature']] ** transf)
                X_trans[state['feature']] += state['intercept']
                X_trans[state['feature']] = 1 / X_trans[state['feature']]
            else:

                for transf, coef in state['operations']:
                    if coef != 0:
                        X_trans['{}_deg{}'.format(state['base_feature'], transf)] = X[state['base_feature']] ** transf
            X_trans.drop([state['base_feature']], axis=1, inplace=True)

        return X_trans


if __name__ == "__main__":

    df = pd.read_csv(r"./data/housing_train.csv", index_col=0)
    print(df.head())