'''
Author: Jacob Howard-Parker
Date: 02/06/2021

Model training for deployment project, IRIS prediction.
'''

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import numpy as np
import pickle

data = load_iris(as_frame=True)

X = data['data']
y = data['target']

estimator = LogisticRegression(penalty='none')
model = OneVsRestClassifier(estimator, n_jobs=-1)

model.fit(X, y)

with open('model/iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

