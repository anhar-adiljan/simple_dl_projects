# import the necessary packages
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from keras.models import Sequential

class LeNet:
    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):
        # initialize the model
        model = Sequential()
        # first set of CONV => RELU => POOL
        model.add(Conv2D(20, (5,5), padding='same', 
            input_shape=(height, width, depth)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # second set of CONV => RELU => POOL
        model.add(Conv2D(50, (5,5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        # load the weights of a pre-trained model if a path is supplied
        if weightsPath is not None:
            model.load_weights(weightsPath)
        # return the constructed network architecture
        return model
