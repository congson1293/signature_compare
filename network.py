from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout, Input

def building_network():
    model = Sequential()

    model.add(Input((150,200,3)))

    model.add(Convolution2D(filters=32,
                            kernel_size=(2,3),
                            padding='valid',
                            activation='relu',
                            strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 1),
                           padding='valid'))
    model.add(Dropout(0.2))

    model.add(Convolution2D(filters=128,
                            kernel_size=(2, 3),
                            padding='valid',
                            activation='relu',
                            strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 1),
                           padding='valid'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model