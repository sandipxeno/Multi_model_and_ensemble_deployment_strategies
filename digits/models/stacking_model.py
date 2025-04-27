import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

class StackingModel:
    def __init__(self, model_paths):
        self.models = [tf.keras.models.load_model(path) for path in model_paths]
        self.meta_model = LogisticRegression(max_iter=1000)

    def fit_meta_model(self, x_val, y_val):
        features = [model.predict(x_val) for model in self.models]
        features = np.concatenate(features, axis=1)
        self.meta_model.fit(features, y_val)

    def predict(self, x):
        features = [model.predict(x) for model in self.models]
        features = np.concatenate(features, axis=1)
        final_preds = self.meta_model.predict(features)
        return final_preds
