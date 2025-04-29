from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

# Load stacking data
X_stack = np.load('D:/Prodigal-5/Models/stacking/stacking_features.npy')
y_stack = np.load('D:/Prodigal-5/Models/stacking/stacking_labels.npy')

# Train meta-learner
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(X_stack, y_stack)

# Save meta-model
joblib.dump(meta_model, 'stacking_meta_model.pkl')

print("Stacking meta-model saved as stacking_meta_model.pkl")
