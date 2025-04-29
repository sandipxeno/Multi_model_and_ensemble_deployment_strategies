# main.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

from bagging.bagging import build_bagging_model, save_bagging_model
from boosting.boosting import build_boosting_model, save_boosting_model
from stacking.stacking import build_stacking_model, save_stacking_model

DATA_DIR = './lion_cheetah_dataset'
MODEL_DIR = './models'
IMG_SIZE = (64, 64)

def load_data():
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=10000,
        class_mode='sparse'
    )
    X, y = next(generator)
    return X, y

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Bagging Model...")
    bagging_model = build_bagging_model(X_train, y_train)
    save_bagging_model(bagging_model, os.path.join(MODEL_DIR, 'bagging_model.pkl'))

    print("Training Boosting Model...")
    boosting_model = build_boosting_model(X_train, y_train)
    save_boosting_model(boosting_model, os.path.join(MODEL_DIR, 'boosting_model.pkl'))

    print("Training Stacking Model...")
    stacking_model = build_stacking_model(X_train, y_train)
    save_stacking_model(stacking_model, os.path.join(MODEL_DIR, 'stacking_model.pkl'))

    print("All models trained and saved!")

if __name__ == '__main__':
    main()
