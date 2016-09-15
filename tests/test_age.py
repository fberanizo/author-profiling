# -*- coding: utf-8 -*- 

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
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

        # Divides in groups of 24-, 25-34, 35-49, 50+
        #func = np.vectorize(lambda age: 0 if age <= 24 else 1 if age <= 34 else 2 if age <= 49 else 3)

        # Divides in groups of 18-24, 25-34, 35-49, 50-64, 65-xx. (PAN 2016)
        func = np.vectorize(lambda age: 0 if age <= 24 else 1 if age <= 34 else 2 if age <= 49 else 3)

        # Divides in groups of 20-, 21-24, 25-34, 35+
        #func = np.vectorize(lambda age: 0 if age <= 20 else 1 if age <= 24 else 2 if age <= 34 else 3)

        # Divides in groups of 18-24, 24-xx.
        #func = np.vectorize(lambda age: 0 if age <= 24 else 1)
        y = func(y)

        #print(np.bincount(y).sum())
        #print(np.bincount(y)/float(np.bincount(y).sum()))
        #sys.exit()

        # Normalize features
        X = normalize(X, norm='l2', axis=0)

        # Applies dimenstionality reduction to dataset
        #pca = PCA(n_components=3)
        #X_new = pca.fit_transform(X)

        evaluation = Evaluation()

        # Most Frequent Class Classifier
        most_frequent = DummyClassifier(strategy='most_frequent')
        (targets, accuracy, precision, recall, f1_score) = evaluation.run(most_frequent, dict(), X, y)
        f = open('output/age.mostfrequent.out', 'a')
        for t, a, p, r, f1 in zip(targets, accuracy, precision, recall, f1_score):
            f.write("%s,%.2f,%.2f,%.2f,%.2f\n" % (t, a, p, r, f1))
        f.close()

        # Evaluates K-Neighbors classifier
        k_neighboors = KNeighborsClassifier()
        n_neighbors = [3, 5, 11, 21, 31]
        (targets, accuracy, precision, recall, f1_score) = evaluation.run(k_neighboors, dict(n_neighbors=n_neighbors), X, y)
        f = open('output/age.knn.out', 'a')
        for t, a, p, r, f1 in zip(targets, accuracy, precision, recall, f1_score):
            f.write("%s,%.2f,%.2f,%.2f,%.2f\n" % (t, a, p, r, f1))
        f.close()

        # Evaluates Random Forest classifier
        random_forest = RandomForestClassifier()
        n_estimators = [2, 3, 5, 10, 20, 40, 60]
        (targets, accuracy, precision, recall, f1_score) = evaluation.run(random_forest, dict(n_estimators=n_estimators), X, y)
        f = open('output/age.randomforest.out', 'a')
        for t, a, p, r, f1 in zip(targets, accuracy, precision, recall, f1_score):
            f.write("%s,%.2f,%.2f,%.2f,%.2f\n" % (t, a, p, r, f1))
        f.close()

        # Evaluates MLP classifier
        mlp = MLPClassifier()
        hidden_layer_sizes = [20, 30, 50, 75, 100, 120, 150]
        activation = ['logistic', 'tanh', 'relu']
        (targets, accuracy, precision, recall, f1_score) = evaluation.run(mlp, dict(hidden_layer_sizes=hidden_layer_sizes, activation=activation), X, y)
        f = open('output/age.mlp.out', 'a')
        for t, a, p, r, f1 in zip(targets, accuracy, precision, recall, f1_score):
            f.write("%s,%.2f,%.2f,%.2f,%.2f\n" % (t, a, p, r, f1))
        f.close()

        # Evaluates SVM
        svm = SVC()
        kernel = ['linear', 'rbf']
        Cs = np.logspace(-3, 4, 8) # C = [0.001, 0.01, .., 1000, 10000]
        gamma = np.logspace(-3, 4, 8) # gamma = [0.001, 0.01, .., 1000, 10000]
        (targets, accuracy, precision, recall, f1_score) = evaluation.run(svm, dict(kernel=kernel, C=Cs, gamma=gamma), X, y)
        f = open('output/age.svm.out', 'a')
        for t, a, p, r, f1 in zip(targets, accuracy, precision, recall, f1_score):
            f.write("%s,%.2f,%.2f,%.2f,%.2f\n" % (t, a, p, r, f1))
        f.close()

if __name__ == '__main__':
    unittest.main()
