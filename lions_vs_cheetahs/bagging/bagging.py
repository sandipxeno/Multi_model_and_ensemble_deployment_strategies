# bagging/bagging.py
import os
import joblib
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

def build_bagging_model(X, y):
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=5,
        random_state=42
    )
    model.fit(X.reshape(X.shape[0], -1), y)
    return model

def save_bagging_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_bagging_model(path):
    return joblib.load(path)
