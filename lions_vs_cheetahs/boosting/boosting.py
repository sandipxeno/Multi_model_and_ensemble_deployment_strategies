# boosting/boosting.py
import os
import joblib
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def build_boosting_model(X, y):
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=5,
        random_state=42
    )
    model.fit(X.reshape(X.shape[0], -1), y)
    return model

def save_boosting_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_boosting_model(path):
    return joblib.load(path)
