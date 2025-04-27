import numpy as np
import tensorflow as tf

class BaggingModel:
    def __init__(self, model_paths):
        self.models = [tf.keras.models.load_model(path) for path in model_paths]

    def predict(self, x):
        preds = [model.predict(x) for model in self.models]
        preds = np.array(preds)
        avg_preds = np.mean(preds, axis=0)
        final_preds = np.argmax(avg_preds, axis=1)
        return final_preds
