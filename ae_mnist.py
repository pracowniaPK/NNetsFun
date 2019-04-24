from tensorflow.nn import relu, softmax
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D, Reshape, Flatten
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

from utils import show_grid


def get_conv_ae(n):
    ae_input = Input((28, 28, 1))

    x = Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation=relu, 
               padding='same')(ae_input)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation=relu, padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    encoder = Dense(n, activation=relu)(x)

    x = Dense(784, activation=relu)(encoder)
    x = Reshape((7,7,16))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation=relu, padding='same')(x)
    x = Conv2D(16, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoder = Conv2D(1, (3, 3), activation=relu, padding='same')(x)

    model = Model(ae_input, decoder)

    model.compile(optimizer='adadelta', loss='binary_crossentropy')

    return model, ae_input, decoder

def get_dense_ae(n):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=relu),
        Dense(64, activation=relu),
        Dense(n, activation=relu),
        Dense(64, activation=relu),
        Dense(128, activation=relu),
        Dense(28*28, activation=softmax),
        Reshape((28, 28)),
    ])

    model.compile(optimizer='adadelta', loss='binary_crossentropy')

    return model

if __name__ == "__main__":
    (train_in, _), (test_in, test_out) = mnist.load_data()
    train_in = train_in / 255.0
    test_in = test_in / 255.0
    train_in = train_in.reshape((train_in.shape[0], 28, 28, 1))
    test_in = test_in.reshape((test_in.shape[0], 28, 28, 1))

    n = 16
    model, _, _ = get_conv_ae(n)
    model.summary()

    x_size = 10
    y_size = 4
    x_epochs = 1
    indices = [np.where(test_out == i)[0][0] for i in range(x_size)]
    fig = plt.figure(figsize=(x_size, y_size))
    out_vis = []
    for i in indices:
        out_vis.append(test_in[i].reshape(28,28))
    for i in range(y_size):
        if i > 0:
            model.fit(train_in, train_in, epochs=x_epochs)
        
        for j in indices:
            out_vis.append(
                model.predict(test_in[j].reshape(1,28,28,1)).reshape(28,28))

    show_grid(np.array(out_vis), x_size, y_size+1, 
              'out\\ae_n{}_ep{}.png'.format(n, x_epochs*y_size))
