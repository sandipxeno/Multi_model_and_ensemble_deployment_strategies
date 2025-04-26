import pickle
import os
from bagging_models import train_bagging_models
from boosting_models import train_boosting_models
from stacking_models import train_stacking_model

# Save all models function
def save_all_models():
    # Create the model folder if it doesn't exist
    if not os.path.exists('model'):
        os.makedirs('model')

    # Train the models
    bagging_models, bagging_scores = train_bagging_models()
    boosting_models, boosting_scores = train_boosting_models()
    stacking_model, stacking_score = train_stacking_model()

    # Combine all models and scores
    all_models = bagging_models + boosting_models + [stacking_model]
    all_scores = bagging_scores + boosting_scores + [stacking_score]

    # Print scores of all models
    print("All model scores:")
    for i, score in enumerate(all_scores):
        model_name = type(all_models[i]).__name__
        print(f"{model_name}: {score:.4f}")

    # Save all models to the 'model' folder
    for i, model in enumerate(all_models):
        model_name = type(model).__name__
        if model_name == "BaggingRegressor":
            # Add the iteration index to the filename for BaggingRegressor
            filename = f'model/BaggingRegressor_{i+1}_model.pkl'
        else:
            filename = f'model/{model_name}_model.pkl'
        
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Saved {model_name} as {filename}")

# Run the script to save models
if __name__ == '__main__':
    save_all_models()