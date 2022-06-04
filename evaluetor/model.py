import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

class Model():
    def __init__(self, cls, metrics, params, name, gss=None, n_jobs=-1, n_iter=50, verbose=0, train_size=.8, random_state=42,
                 scoring="balanced_accuracy", grid_search=True):
        self.cls = cls
        self.metrics = metrics
        self.params = params
        self.n_jobs = n_jobs
        self.gss = gss
        self.name = name
        self.n_iter = n_iter
        self.verbose = verbose
        self.train_size = train_size
        self.random_state = random_state
        self.scoring = scoring
        self.grid_search = grid_search


    def eval(self, X_train, X_test, y_train, y_test, group_train=None):
        if self.gss:
            cv = self.gss
        else:
            cv = GroupShuffleSplit(n_splits=5, train_size=self.train_size, random_state=self.random_state)

        if not self.grid_search:
            search = RandomizedSearchCV(self.cls, param_distributions=self.params, n_iter=self.n_iter,
                                        scoring=self.scoring, n_jobs=self.n_jobs, cv=cv, verbose=self.verbose,
                                        random_state=self.random_state)
        else:
            search = GridSearchCV(self.cls, param_grid=self.params, scoring=self.scoring,
                                  n_jobs=self.n_jobs, cv=cv, verbose=self.verbose)

        if group_train:
            search.fit(X=X_train, y=y_train, groups=group_train)
        else:
            search.fit(X=X_train, y=y_train)

        cls = search.best_estimator_
        # make predictions for test data
        yy_train = cls.predict(X_train)
        yy_train = np.round(yy_train)
        yy = cls.predict(X_test)
        yy = np.round(yy)
        results = {}
        for metric in self.metrics:
            results[metric.__name__ + "_train"] = metric(y_train, yy_train)
            results[metric.__name__ + "_test"] = metric(y_test, yy)


        results = pd.DataFrame.from_dict([results]).T.reset_index()
        results.columns = [['metric', 'value']]
        results['classifier'] = self.name
        return results, yy, search
