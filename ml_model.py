#!/usr/bin/python3
# -*- coding: utf-8 -*-
###############################>GENERAL-INFORMATIONS<###############################
"""
Build in Python 3.6

Author:
Filipe Dezordi
zimmer.filipe@gmail.com
https://github.com/dezordi

Dependencies on .yml enviroment file
"""

###############################>LIBRARIES<###############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from mlxtend.plotting import plot_confusion_matrix
###############################>DATA<###############################
data = pd.read_csv('pima-indians-diabetes.csv',header = None)
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

normaliza = MinMaxScaler()
x_normalizadas=normaliza.fit_transform(x)
x_norm_train, x_norm_test, y_train, y_test = train_test_split(x_normalizadas, y,
test_size=0.30,random_state=42)
###############################>CLASSIFIERS<###############################
#KNN
clf_KNN = KNeighborsClassifier(n_neighbors=5)
clf_KNN.fit(x_norm_train, y_train)
y_pred = clf_KNN.predict(x_norm_test)
print("KNN:\n",classification_report(y_test, y_pred))
#RANDOM FOREST
clf_arvore = DecisionTreeClassifier(random_state=1)
clf_arvore = clf_arvore.fit(x_norm_train, y_train)
y_pred = clf_arvore.predict(x_norm_test)
print("Random Forest:\n",classification_report(y_test, y_pred))
#MLP NETWORK
clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 10), random_state=1, max_iter=400)
clf_mlp = clf_mlp.fit(x_norm_train, y_train)
y_pred = clf_mlp.predict(x_norm_test)
print("MLP:\n",classification_report(y_test, y_pred))

###############################>SAVING BEST MODEL<###############################
nome_do_arquivo = 'melhor_modelo.sav'
joblib.dump(clf_mlp,nome_do_arquivo)
modelo_salvo = joblib.load(nome_do_arquivo)