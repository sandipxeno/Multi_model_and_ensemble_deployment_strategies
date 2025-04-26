# bagging_models.py
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from data_preprocessing import load_and_preprocess_data

def train_bagging_models():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    models = []
    scores = []

    for i in range(3):
        base_model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=i)
        bagging = BaggingRegressor(base_estimator=base_model, n_estimators=5, random_state=i)
        bagging.fit(X_train, y_train)

        score = bagging.score(X_test, y_test)
        scores.append(score)
        models.append(bagging)

    return models, scores
