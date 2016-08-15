# -*- coding: utf-8 -*- 

import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

class Evaluation:
    """Classifier evaluator."""

    def __init__(self):
        pass

    def run(self, classifier, X, y):
        """Performs classifier evaluation."""
        print 'Evaluating ' + type(classifier).__name__
        n_folds = 3
        score_per_fold = []
        C_per_fold = []

        for f, (train_index, test_index) in enumerate(StratifiedKFold(y, n_folds=n_folds)):
            print 'Fold ' + str(f)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            Cs = np.logspace(-3, 4, 8) # C = [0.001, 0.01, .., 1000, 10000]
            skf = StratifiedKFold(y_train, n_folds=n_folds)
            clf = GridSearchCV(estimator=classifier, param_grid=dict(C=Cs), cv=skf, n_jobs=1)
            
            clf.fit(X_train, y_train)
            print 'Training score: ' + str(clf.best_score_)
            print 'Best C: ' + str(clf.best_estimator_.C)

            score = clf.score(X_test, X_test)
            print 'Test score: ' + str(score)

            score_per_fold.append(score)
            C_per_fold.append(clf.best_estimator_.C)

        print 'Average test score: ' + str(np.mean(score_per_fold))

        return np.mean(score_per_fold)
