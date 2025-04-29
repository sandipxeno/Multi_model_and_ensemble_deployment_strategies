import numpy as np
import joblib
from xgboost import XGBClassifier

# Load extracted features
X = np.load('D:/Prodigal-5/Models/boosting/boosting_features.npy')
y = np.load('D:/Prodigal-5/Models/boosting/boosting_labels.npy')

# XGBoost model
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    verbosity=1  # <-- shows training info
)

print("Training XGBoost model...")

# Fit with eval_set for live progress
xgb_model.fit(
    X, y,
    eval_set=[(X, y)],  # you can later add a separate validation set here if you want
    verbose=True  # <-- print after each boosting round
)

# Save model
joblib.dump(xgb_model, 'boosting_xgboost_model.pkl')
print("XGBoost model saved successfully as boosting_xgboost_model.pkl!")
