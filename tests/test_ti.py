# -*- coding: utf-8 -*- 

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest, pandas
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from classifiers.evaluation import Evaluation

class TITestSuite(unittest.TestCase):
    """Tests ti classification."""

    def test_ti(self):
        lexical_features =  pandas.read_csv('input/lexical-features.csv', sep=';')
        subjects = pandas.read_excel('input/subjects.xlsx')

        X = lexical_features.ix[:, lexical_features.columns != 'seq '].as_matrix()
        y = subjects.loc[subjects['id'].isin(lexical_features[' ID '])]['ti'].as_matrix()

        # Applies dimenstionality reduction to dataset
        pca = PCA(n_components=3)
        X_new = pca.fit_transform(X)

        evaluation = Evaluation()

        # Evaluates Linear SVM classifier
        linear_svm = SVC(kernel='linear')
        evaluation.run(linear_svm, X_new[:50,], y[:50,])

        # Evaluates RBF SVM classifier
        # rbf_svm = SVC(kernel='rbf')
        # evaluation.run(linear_svm, X_new[:150,], y[:150,])
        
        # Evaluates K-Neighbors classifier
        # k_neighboors = KNeighborsClassifier()
        # evaluation.run(linear_svm, X_new[:150,], y[:150,])
        
        # Evaluates Random Forest classifier
        # random_forest = RandomForestClassifier()
        # evaluation.run(linear_svm, X_new[:150,], y[:150,])

if __name__ == '__main__':
    unittest.main()