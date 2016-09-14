# -*- coding: utf-8 -*- 

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from classifiers.evaluation import Evaluation

class ReligiousityTestSuite(unittest.TestCase):
    """Tests religiousity classification."""

    def test_religiousity(self):
        lexical_features =  pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), 'input', 'lexical-features.csv')), sep=';')
        subjects = pd.read_excel(os.path.abspath(os.path.join(os.path.dirname(__file__), 'input', 'subjects.xlsx')))

        X = lexical_features.ix[:, lexical_features.columns != 'seq '].sort_values(by=' ID ')
        y = subjects.loc[subjects['id'].isin(lexical_features[' ID '])].sort_values(by='id')['religiousity']

        # Removes ID column from features
        X.drop(' ID ', axis=1, inplace=True)

        # Removes samples without religiousity value or religiousity = 3
        y = y.replace({3.0: None})
        X = X.loc[~y.apply(pd.isnull)].as_matrix()
        y = y[~y.apply(pd.isnull)].as_matrix()

        # Transform output into binary values
        func = np.vectorize(lambda religiousity: 0 if religiousity < 3 else 1)
        y = func(y)
        lb = LabelBinarizer()
        y = lb.fit_transform(y)[:,0]

        # Normalize features
        X = normalize(X, norm='l2', axis=0)

        # Applies dimenstionality reduction to dataset
        #pca = PCA(n_components=3)
        #X_new = pca.fit_transform(X)

        evaluation = Evaluation()

        # Most Frequent Class Classifier
        most_frequent = DummyClassifier(strategy='most_frequent')
        (accuracy, precision, recall, f1_score) = evaluation.run(most_frequent, dict(), X, y)
        f = open('output/religiousity.mostfrequent.out', 'a')
        f.write("%.2f,%.2f,%.2f,%.2f\n" % (accuracy, precision, recall, f1_score))
        f.close()

        # Evaluates K-Neighbors classifier
        k_neighboors = KNeighborsClassifier()
        n_neighbors = [3, 5, 11, 21, 31]
        (accuracy, precision, recall, f1_score) = evaluation.run(k_neighboors, dict(n_neighbors=n_neighbors), X, y)
        f = open('output/religiousity.knn.out', 'a')
        f.write("%.2f,%.2f,%.2f,%.2f\n" % (accuracy, precision, recall, f1_score))
        f.close()

        # Evaluates Random Forest classifier
        random_forest = RandomForestClassifier()
        n_estimators = [2, 3, 5, 10, 20, 40, 60]
        (accuracy, precision, recall, f1_score) = evaluation.run(random_forest, dict(n_estimators=n_estimators), X, y)
        f = open('output/religiousity.randomforest.out', 'a')
        f.write("%.2f,%.2f,%.2f,%.2f\n" % (accuracy, precision, recall, f1_score))
        f.close()

        # Evaluates MLP classifier
        mlp = MLPClassifier()
        hidden_layer_sizes = [20, 30, 50, 75, 100, 120, 150]
        activation = ['logistic', 'tanh', 'relu']
        (accuracy, precision, recall, f1_score) = evaluation.run(mlp, dict(hidden_layer_sizes=hidden_layer_sizes, activation=activation), X, y)
        f = open('output/religiousity.mlp.out', 'a')
        f.write("%.2f,%.2f,%.2f,%.2f\n" % (accuracy, precision, recall, f1_score))
        f.close()

        # Evaluates Linear SVM classifier
        linear_svm = SVC(kernel='linear')
        Cs = np.logspace(-3, 4, 8) # C = [0.001, 0.01, .., 1000, 10000]
        (accuracy, precision, recall, f1_score) = evaluation.run(linear_svm, dict(C=Cs), X, y)
        f = open('output/religiousity.linsvm.out', 'a')
        f.write("%.2f,%.2f,%.2f,%.2f\n" % (accuracy, precision, recall, f1_score))
        f.close()

        # Evaluates RBF SVM classifier
        rbf_svm = SVC(kernel='rbf')
        Cs = np.logspace(-3, 4, 8) # C = [0.001, 0.01, .., 1000, 10000]
        gamma = np.logspace(-3, 4, 8) # gamma = [0.001, 0.01, .., 1000, 10000]
        (accuracy, precision, recall, f1_score) = evaluation.run(rbf_svm, dict(C=Cs, gamma=gamma), X, y)
        f = open('output/religiousity.rbfsvm.out', 'a')
        f.write("%.2f,%.2f,%.2f,%.2f\n" % (accuracy, precision, recall, f1_score))
        f.close()

if __name__ == '__main__':
    unittest.main()
