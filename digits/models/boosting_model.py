import numpy as np
import tensorflow as tf

class BoostingModel:
    def __init__(self, model_paths, weights=None):
        self.models = [tf.keras.models.load_model(path) for path in model_paths]
        self.weights = weights if weights else [1.0/len(self.models)] * len(self.models)

    def predict(self, x):
        preds = [model.predict(x) * weight for model, weight in zip(self.models, self.weights)]
        preds = np.array(preds)
        weighted_sum = np.sum(preds, axis=0)
        final_preds = np.argmax(weighted_sum, axis=1)
        return final_preds
