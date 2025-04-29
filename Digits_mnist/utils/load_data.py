import tensorflow as tf

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., None].astype('float32') / 255.0  # Add channel dimension
    x_test = x_test[..., None].astype('float32') / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return (x_train, y_train), (x_test, y_test)
