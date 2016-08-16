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
        n_folds = 3

        skf = StratifiedKFold(n_folds=n_folds, shuffle=True)
        for train_index, test_index in skf.split(X, y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            scores = ['accuracy', 'precision', 'recall']

            for score in scores:
                print("# Tuning hyper-parameters for %s" % score)
                print("")

                clf = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring=score, verbose=1, n_jobs=2)
                clf.fit(X_train, y_train)

                print("Best parameters set found on validation set:")
                print("")
                print(clf.best_params_)
                print("")
                print("Grid scores on validation set:")
                print("")
                results = dict(filter(lambda i:i[0] in ["params", "test_mean_score", "test_std_score"], clf.results_.iteritems()))
                table = PrettyTable()
                for key, val in sorted(results.iteritems()):
                  table.add_column(key, sorted(val))
                print table
                #for params, mean_score, scores in clf.results_:
                #    print("%0.3f (+/-%0.03f) for %r"
                #          % (mean_score, scores.std() * 2, params))
                print("")
                print("Scores on test set:")
                print("")
                y_true, y_pred = y_test, clf.predict(X_test)
                print classification_report(y_true, y_pred)
