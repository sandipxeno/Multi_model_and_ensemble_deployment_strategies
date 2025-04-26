# boosting_models.py
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from data_preprocessing import load_and_preprocess_data

def train_boosting_models():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    models = [
        XGBRegressor(n_estimators=100, random_state=0),
        LGBMRegressor(n_estimators=100, random_state=1),
        CatBoostRegressor(verbose=0, random_state=2)
    ]

    trained_models = []
    scores = []

    for model in models:
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        trained_models.append(model)

    return trained_models, scores
