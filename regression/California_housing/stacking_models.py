# stacking_models.py
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from data_preprocessing import load_and_preprocess_data

def train_stacking_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    base_models = [
        ('mlp1', MLPRegressor(hidden_layer_sizes=(64,), max_iter=300, random_state=0)),
        ('mlp2', MLPRegressor(hidden_layer_sizes=(128,), max_iter=300, random_state=1)),
        ('mlp3', MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, random_state=2)),
    ]

    meta_model = Ridge()

    stacking = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    stacking.fit(X_train, y_train)
    score = stacking.score(X_test, y_test)

    return stacking, score
