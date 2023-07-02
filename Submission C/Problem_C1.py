# =============================================================================
# PROBLEM C1
#
# Given two arrays, train a neural network model to match the X to the Y.
# Predict the model with new values of X [-2.0, 10.0]
# We provide the model prediction, do not change the code.
#
# The test infrastructure expects a trained model that accepts
# an input shape of [1]
# Do not use lambda layers in your model.
#
# Please be aware that this is a linear model.
# We will test your model with values in a range as defined in the array to make sure your model is linear.
#
# Desired loss (MSE) < 1e-4
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def solution_C1():
    # DO NOT CHANGE THIS CODE
    X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    Y = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    # YOUR CODE HERE
    normalizer = tf.keras.layers.Normalization(axis=None, input_shape=(1,))
    normalizer.adapt(X)
    # CREATE CUSTOM CALLBACK
    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('loss') < 1e-4):
                print('\nMSE Loss is lesser than 1e-4 !')
                print('\nStop training model...')
                self.model.stop_training = True

    # MAKE VARIABLE FOR CUSTOM CALLBACK
    custom_callback = MyCallback()

    model = Sequential()
    model.add(normalizer)
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dense(1))


    model.compile(optimizer='adam',
                  metrics='accuracy',
                  loss='mean_squared_error')

    model.fit(X, Y, epochs=1000, callbacks=[custom_callback])
    print(model.predict([-2.0, 10.0]))
    return model


# The code below is to save your model as a .h5 file
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C1()
    model.save("model_C1.h5")
