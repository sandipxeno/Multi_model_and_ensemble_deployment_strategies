# stacking/stacking.py
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def build_stacking_model(X, y):
    # Flatten the images for scikit-learn models
    X_flat = X.reshape(X.shape[0], -1)
    
    # Define base models
    base_models = [
        ('dt', DecisionTreeClassifier(max_depth=3)),
        ('rf', RandomForestClassifier(n_estimators=5))
    ]
    
    # Define meta model with increased max_iter
    meta_model = LogisticRegression(max_iter=500)  # <-- Fixed here
    
    # Build stacking model
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model
    )
    
    # Train stacking model
    stacking_model.fit(X_flat, y)
    
    return stacking_model

def save_stacking_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_stacking_model(path):
    return joblib.load(path)
