# -*- coding: utf-8 -*- 

import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

class Evaluation:
    """Classifier evaluator."""

    def __init__(self):
        pass

    def run(self, classifier, param_grid, X, y):
        """Performs classifier evaluation."""
        print('Evaluating ' + type(classifier).__name__)
        n_folds = 5

        for fold, (train_index, test_index) in enumerate(StratifiedKFold(y, n_folds=n_folds)):
            print('Fold %d' % fold)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            scores = ['precision', 'recall']

            for score in scores:
                print("# Tuning hyper-parameters for %s" % score)
                print("")

                skf = StratifiedKFold(y_train, n_folds=n_folds)
                clf = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=skf, n_jobs=4, scoring='%s_weighted' % score)
                clf.fit(X_train, y_train)

                print("Best parameters set found on validation set:")
                print("")
                print(clf.best_params_)
                print("")
                print("Grid scores on validation set:")
                print("")
                for params, mean_score, scores in clf.grid_scores_:
                    print("%0.3f (+/-%0.03f) for %r"
                          % (mean_score, scores.std() * 2, params))
                print("")
                print("Scores on test set:")
                print("")
                y_true, y_pred = y_test, clf.predict(X_test)
                print classification_report(y_true, y_pred)
