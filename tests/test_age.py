# -*- coding: utf-8 -*- 

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from classifiers.evaluation import Evaluation

class AgeTestSuite(unittest.TestCase):
    """Tests age classification."""

    def test_age(self):
        lexical_features =  pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), 'input', 'lexical-features.csv')), sep=';')
        subjects = pd.read_excel(os.path.abspath(os.path.join(os.path.dirname(__file__), 'input', 'subjects.xlsx')))

        X = lexical_features.ix[:, lexical_features.columns != 'seq '].sort_values(by=' ID ')
        y = subjects.loc[subjects['id'].isin(lexical_features[' ID '])].sort_values(by='id')['age']

        # Removes ID column from features
        X.drop(' ID ', axis=1, inplace=True)
        
        # Removes samples without age value
        X = X.loc[~y.apply(pd.isnull)].as_matrix()
        y = y[~y.apply(pd.isnull)].as_matrix()

        # Divides in groups of 20-, 21-25, 26-30, 31-45, 45+
        #func = np.vectorize(lambda age: 0 if age < 21 else 1 if age < 26 else 2 if age < 31 else 3 if age < 45 else 4)

        # Divides in groups of 18-24, 25-34, 35-49, 50-64, 65-xx. (PAN 2016)
        func = np.vectorize(lambda age: 0 if age <= 24 else 1 if age <= 2 else 34 if age <= 49 else 3 if age <= 64 else 4)

        # Divides in groups of 18-24, 24-xx.
        #func = np.vectorize(lambda age: 0 if age <= 24 else 1)
        y = func(y)

        # Normalize features
        X = normalize(X, norm='l2', axis=0)

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

        # Evaluates MLP classifier
        mlp = MLPClassifier()
        hidden_layer_sizes = [20, 30, 50, 75, 100, 120, 150]
        activation = ['logistic', 'tanh', 'relu']
        evaluation.run(mlp, dict(hidden_layer_sizes=hidden_layer_sizes, activation=activation), X, y)

        # Evaluates Linear SVM classifier
        linear_svm = SVC(kernel='linear')
        Cs = np.logspace(-3, 4, 8) # C = [0.001, 0.01, .., 1000, 10000]
        evaluation.run(linear_svm, dict(C=Cs), X, y)

        # Evaluates RBF SVM classifier
        rbf_svm = SVC(kernel='rbf')
        Cs = np.logspace(-3, 4, 8) # C = [0.001, 0.01, .., 1000, 10000]
        gamma = np.logspace(-3, 4, 8) # gamma = [0.001, 0.01, .., 1000, 10000]
        evaluation.run(rbf_svm, dict(C=Cs, gamma=gamma), X, y)

if __name__ == '__main__':
    unittest.main()
