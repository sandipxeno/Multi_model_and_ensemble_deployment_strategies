import tensorflow as tf
from load_data import load_mnist_data

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_models():
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    for i in range(3):
        model = build_model()
        print(f"Training model {i+1}...")
        model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
        model.save(f'models/cnn_model_{i}.h5')

if __name__ == "__main__":
    train_and_save_models()

