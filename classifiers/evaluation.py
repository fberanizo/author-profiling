# -*- coding: utf-8 -*- 

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable

class Evaluation:
    """Classifier evaluator."""

    def __init__(self):
        pass

    def run(self, classifier, param_grid, X, y):
        """Performs classifier evaluation."""
        print('Evaluating ' + type(classifier).__name__)

        # Split the dataset in two equal parts
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        scores = ['f1'] if len(set(y_test)) <= 2 else ['f1_weighted']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print("")

            skf = StratifiedKFold(n_splits=10)
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
            print(classification_report(y_true, y_pred))

            avg_accuracy = accuracy_score(y_true, y_pred)

            print("")
            print("Average accuracy on test set (using best parameters): %.2f" % avg_accuracy)
            print("")

            (precision, recall, f1_score, support) = precision_recall_fscore_support(y_true, y_pred)
            print("===================================================================")
            print(precision)
            print("===================================================================")
            average = 'binary' if len(set(y_pred)) <= 2 else 'weighted'
            (avg_precision, avg_recall, avg_f1_score, avg_support) = precision_recall_fscore_support(y_true, y_pred, average=average)

            accuracy = []
            for target in target_names:
                accuracy.append(accuracy_score(y_true[np.where(y_true == np.int_(target))[0]], y_pred[np.where(y_true == np.int_(target))[0]]))

            target_names.append('avg')
            accuracy = np.append(accuracy, avg_accuracy)
            precision = np.append(precision, avg_precision)
            recall = np.append(recall, avg_recall)
            f1_score = np.append(f1_score, avg_f1_score)

            return target_names, accuracy, precision, recall, f1_score
            
