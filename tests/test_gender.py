# -*- coding: utf-8 -*- 

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from classifiers.evaluation import Evaluation

class GenderTestSuite(unittest.TestCase):
    """Tests gender classification."""

    def test_gender(self):
        lexical_features =  pd.read_csv('input/lexical-features.csv', sep=';')
        subjects = pd.read_excel('input/subjects.xlsx')

        X = lexical_features.ix[:, lexical_features.columns != 'seq ']
        y = subjects.loc[subjects['id'].isin(lexical_features[' ID '])]['gender']

        # Removes samples without gender value
        X = X[~y.apply(pd.isnull)].as_matrix()
        y = y[~y.apply(pd.isnull)].as_matrix()

        # Applies dimenstionality reduction to dataset
        #pca = PCA(n_components=3)
        #X_new = pca.fit_transform(X)

        evaluation = Evaluation()

        # Evaluates K-Neighbors classifier
        k_neighboors = KNeighborsClassifier()
        n_neighbors = [3, 5, 11, 21, 31]
        evaluation.run(k_neighboors, dict(n_neighbors=n_neighbors), X, y)

        # Evaluates Random Forest classifier
        random_forest = RandomForestClassifier()
        n_estimators = [2, 3, 5, 10, 20, 40, 60]
        evaluation.run(random_forest, dict(n_estimators=n_estimators), X, y)

        # Evaluates Linear SVM classifier
        linear_svm = SVC(kernel='linear')
        Cs = np.logspace(-3, 4, 8) # C = [0.001, 0.01, .., 1000, 10000]
        evaluation.run(linear_svm, dict(C=Cs), X, y)

        # Evaluates RBF SVM classifier
        rbf_svm = SVC(kernel='rbf')
        Cs = np.logspace(-3, 4, 8) # C = [0.001, 0.01, .., 1000, 10000]
        evaluation.run(rbf_svm, dict(C=Cs), X, y)

if __name__ == '__main__':
    unittest.main()