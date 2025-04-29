import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import joblib
from sklearn.linear_model import LogisticRegression
from bagging import LightCNN
from smallCNN_stacking import SmallCNN
import torch.nn as nn

# Load all pre-trained models (Bagging, Boosting, Stacking)
def load_models():
    bagging_models = []
    for i in range(5):
        model = LightCNN(num_classes=10).to('cpu')
        try:
            model.load_state_dict(torch.load(f'D:/Prodigal-5/Models/bagging/bagging_model_{i}.pth', map_location=torch.device('cpu')))
        except FileNotFoundError:
            print(f"Error: Bagging model file not found: D:/Prodigal-5/Models/bagging/bagging_model_{i}.pth")
            return None, None, None, None, None
        model.eval()
        bagging_models.append(model)

    try:
        boosting_model = joblib.load('D:/Prodigal-5/Models/boosting/boosting_xgboost_model.pkl')
    except FileNotFoundError:
        print("Error: Boosting model file not found: D:/Prodigal-5/Models/boosting/boosting_xgboost_model.pkl")
        return None, None, None, None, None

    stacking_cnn = SmallCNN().to('cpu')
    try:
        stacking_cnn.load_state_dict(torch.load('D:/Prodigal-5/Models/stacking/stacking_smallcnn.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        print("Error: Stacking CNN model file not found: D:/Prodigal-5/Models/stacking/stacking_smallcnn.pth")
        return None, None, None, None, None
    stacking_cnn.eval()

    try:
        meta_model = joblib.load('D:/Prodigal-5/Models/stacking/stacking_meta_model.pkl')
    except FileNotFoundError:
        print("Error: Stacking meta model file not found: D:/Prodigal-5/Models/stacking/stacking_meta_model.pkl")
        return None, None, None, None, None

    # CIFAR-10 Class Names
    class_names = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

    return bagging_models, boosting_model, stacking_cnn, meta_model, class_names


# Image Preprocessing function (CRITICAL: Resize to 32x32)
def preprocess_image(image_path):
    target_size = (32, 32)  # MATCH THE TRAINING SIZE OF LightCNN
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    image = transform(image)
    image = image.unsqueeze(0)
    return image


# Bagging prediction
def bagging_predict(models, image_tensor, class_names):
    if image_tensor is None:
        return None, None
    predictions = []
    for model in models:
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted_class = torch.max(output, 1)
            predictions.append(predicted_class.item())

    if predictions:
        final_prediction_index = max(set(predictions), key=predictions.count)
        final_prediction_name = class_names.get(final_prediction_index, f"Unknown Class {final_prediction_index}")
        return final_prediction_name, predictions  # Return the actual class name
    else:
        return None, None


# Boosting prediction
def boosting_predict(model, image_tensor, class_names):
    if image_tensor is None:
        return None, None
    image_flat = image_tensor.view(1, -1).numpy()
    try:
        prediction = model.predict(image_flat)
        prediction_name = class_names.get(prediction[0], f"Unknown Class {prediction[0]}")
        return prediction_name, None
    except Exception as e:
        print(f"Error during boosting prediction: {e}")
        return None, None


# Stacking prediction
def stacking_predict(bagging_models, stacking_cnn, boosting_model, meta_model, image_tensor, class_names):
    if image_tensor is None:
        return None, None
    with torch.no_grad():
        try:
            lightcnn_features = bagging_models[0].features(image_tensor)
            lightcnn_features_flat = torch.flatten(lightcnn_features, 1).cpu().numpy()
        except AttributeError as e:
            print(f"Error accessing LightCNN features: {e}")
            return None, None
        except Exception as e:
            print(f"Error during LightCNN feature extraction: {e}")
            return None, None

        try:
            smallcnn_features = stacking_cnn.conv(image_tensor)
            smallcnn_features_flat = torch.flatten(smallcnn_features, 1).cpu().numpy()
        except AttributeError as e:
            print(f"Error accessing SmallCNN conv layers: {e}")
            return None, None
        except Exception as e:
            print(f"Error during SmallCNN feature extraction: {e}")
            return None, None

    boosting_prediction, _ = boosting_predict(boosting_model, image_tensor, class_names)
    boosting_prediction_index = None
    if boosting_prediction is not None:
        # Need to get the numerical index back from the name if needed for meta-model training context
        for key, value in class_names.items():
            if value == boosting_prediction[0]:
                boosting_prediction_index = key
                break
        boosting_feature = np.array([boosting_prediction_index])
    else:
        boosting_feature = np.array([-1]) # Or some other placeholder

    stacked_features = np.concatenate([lightcnn_features_flat, smallcnn_features_flat, boosting_feature], axis=1)

    try:
        meta_prediction_index = meta_model.predict(stacked_features)[0]
        meta_prediction_name = class_names.get(meta_prediction_index, f"Unknown Class {meta_prediction_index}")
        return meta_prediction_name, None
    except ValueError as e:
        print(f"Error during meta model prediction: {e}")
        print(f"Shape of stacked_features: {stacked_features.shape}")
        return None, None


# A/B Testing + Fallback Mechanism
def ab_testing(bagging_models, boosting_model, stacking_cnn, meta_model, image_tensor, class_names):
    bagging_prediction, _ = bagging_predict(bagging_models, image_tensor, class_names)

    if bagging_prediction is not None:
        return bagging_prediction, None
    else:
        boosting_prediction, _ = boosting_predict(boosting_model, image_tensor, class_names)

        if boosting_prediction is not None:
            return boosting_prediction, None
        else:
            stacking_prediction, _ = stacking_predict(bagging_models, stacking_cnn, boosting_model, meta_model, image_tensor, class_names)
            return stacking_prediction, None


# Low Latency Ensemble Aggregation
def low_latency_ensemble(bagging_models, boosting_model, stacking_cnn, meta_model, image_tensor, class_names):
    bagging_prediction, _ = bagging_predict(bagging_models, image_tensor, class_names)
    boosting_prediction, _ = boosting_predict(boosting_model, image_tensor, class_names)
    stacking_prediction, _ = stacking_predict(bagging_models, stacking_cnn, boosting_model, meta_model, image_tensor, class_names)

    predictions = [result[0] for result in [bagging_prediction, boosting_prediction, stacking_prediction] if isinstance(result, tuple) and result[0] is not None]

    all_predictions_with_none = [bagging_prediction[0] if isinstance(bagging_prediction, tuple) else None,
                                  boosting_prediction[0] if isinstance(boosting_prediction, tuple) else None,
                                  stacking_prediction[0] if isinstance(stacking_prediction, tuple) else None]

    if predictions:
        final_prediction = max(set(predictions), key=predictions.count)
        return final_prediction, all_predictions_with_none
    else:
        return None, None
