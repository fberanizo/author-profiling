# -*- coding: utf-8 -*- 

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from prettytable import PrettyTable

class Evaluation:
    """Classifier evaluator."""

    def __init__(self):
        pass

    def run(self, classifier, param_grid, X, y):
        """Performs classifier evaluation."""
        print('Evaluating ' + type(classifier).__name__)
        n_splits = 5

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        for train_index, test_index in skf.split(X, y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            scores = ['f1_weighted']

            for score in scores:
                print("# Tuning hyper-parameters for %s" % score)
                print("")

                clf = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring=score, cv=skf, verbose=1, n_jobs=2)
                clf.fit(X_train, y_train)

                print("Grid scores on validation set:")
                print("")
                results = dict(filter(lambda i:i[0] in ["params", "test_mean_score", "test_std_score", "test_rank_score"], clf.results_.items()))
                table = PrettyTable()
                for key, val in results.items():
                  table.add_column(key, val)
                print(table)
                
                print("Best parameters set found on validation set:")
                print("")
                print(clf.best_params_)
                print("")

                print("")
                print("Scores on test set (using best parameters):")
                print("")
                y_true, y_pred = y_test, clf.predict(X_test)
                target_names = list(map(str, np.unique(y_true).tolist()))
                print(classification_report(y_true, y_pred, target_names=target_names))
