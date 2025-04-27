import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.bagging_model import BaggingModel
from models.boosting_model import BoostingModel
from models.stacking_model import StackingModel
import random
import numpy as np

class EnsembleAggregator:
    def __init__(self):
        self.model_paths = ['models/cnn_model_0.h5', 'models/cnn_model_1.h5', 'models/cnn_model_2.h5']
        self.bagging = BaggingModel(self.model_paths)
        self.boosting = BoostingModel(self.model_paths)
        self.stacking = StackingModel(self.model_paths)

    def predict(self, x, method='bagging'):
        try:
            if method == 'bagging':
                return self.bagging.predict(x)
            elif method == 'boosting':
                return self.boosting.predict(x)
            elif method == 'stacking':
                return self.stacking.predict(x)
            else:
                raise ValueError("Invalid method.")
        except Exception as e:
            # fallback to bagging if something fails
            print(f"Fallback due to error: {e}")
            return self.bagging.predict(x)

    def a_b_test(self, x):
        method = random.choice(['bagging', 'boosting', 'stacking'])
        print(f"A/B Testing with method: {method}")
        return self.predict(x, method)
