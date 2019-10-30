import pandas as pd
from itertools import combinations, chain
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


class LRFEPipeline(BaseEstimator, TransformerMixin):

    def __init__(self, cv=3, inversion=False, trans_range=None):
        self.cv = cv
        self.inversion = inversion
        self.fit_state = []
        self.trans_range = trans_range if trans_range else [0.5, 1, 2, 3]

    @staticmethod
    def powerset(i):
        s = list(i)  # allows duplicate elements
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    def fit(self, X, y=None):

        self.fit_state = []
        for feat in X.columns:

            if X[feat].dtype not in ['float64']:
                continue

            min_err = 10e10
            X_trial = pd.DataFrame()
            for deg in self.trans_range:
                X_trial[deg] = X[feat] ** deg

            best_inv_state = 0
            best_combo = [1.0]
            for inv_state, target in enumerate([y, 1 / y]):

                if inv_state != 0 and self.inversion == False:
                    break

                for i, combo in enumerate(self.powerset(X_trial.columns), 1):

                    combo = list(combo)
                    model = LinearRegression if len(combo) == 1 else Lasso

                    kf = KFold(n_splits=self.cv, shuffle=True)
                    err = []
                    for train_index, test_index in kf.split(X_trial):
                        X_train, X_test = X_trial[combo].iloc[train_index], X_trial[combo].iloc[test_index]
                        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
                        pd.DataFrame(y_train).to_csv(r"C:\Users\olive\Desktop\Book1.csv")
                        md = model().fit(X_train, y_train)
                        y_pred = md.predict(X_test)
                        err.append(mean_squared_error(y_test, y_pred))

                    avg_err = sum(err) / len(err)
                    if avg_err < min_err:
                        min_err = avg_err
                        best_combo, best_inv_state = combo, inv_state

            target = 1 / y if best_inv_state else y
            model = LinearRegression if len(best_combo) == 1 else Lasso
            md = model().fit(X_trial[best_combo], target)
            operations = [(transf, coef) for transf, coef in zip(best_combo, md.coef_) if coef != 0]
            intercept = md.intercept_

            if best_inv_state:
                feature_str = ' + '.join(
                    ['{} * {} ^ {}'.format(coef, feat, transf)
                     if transf != 1.0 else feat for transf, coef in zip(best_combo, md.coef_) if coef != 0])
                feature_str = '{} + {}'.format(feature_str, intercept)
                feature_str = '1 / ({})'.format(feature_str)
            else:
                feature_str = feat

            d = {'feature': feature_str, 'base_feature': feat, 'operations': operations,
                 'intercept': intercept, 'inv_state': best_inv_state}
            self.fit_state.append(d)

        return self

    def transform(self, X, y=None):

        X_trans = X.copy()
        for state in self.fit_state:

            if state['inv_state'] == 1:
                X_trans[state['feature']] = 0.0
                for transf, coef in state['operations']:
                    if coef != 0: X_trans[state['feature']] += coef * (X[state['base_feature']] ** transf)
                X_trans[state['feature']] += state['intercept']
                X_trans[state['feature']] = 1 / X_trans[state['feature']]
            else:
                for transf, coef in state['operations']:
                    if coef != 0 and transf != 1:
                        X_trans['{}_deg{}'.format(state['base_feature'], transf)] = \
                            X[state['base_feature']] ** transf

            if 1.0 not in [x[0] for x in state['operations']]:
                X_trans.drop([state['base_feature']], axis=1, inplace=True)

        return X_trans
