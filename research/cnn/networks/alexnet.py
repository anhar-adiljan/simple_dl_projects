# import the necessary packages
from keras.layers import BatchNormalization
from keras.layers import ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense

class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):
        # initialize the model
        model = Sequential()
        # CONV1 => POOL1 => NORM1 layers
        model.add(Conv2D(96, (11, 11), padding='valid', activation='relu', strides=4, input_shape=(height, width, depth)))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(BatchNormalization())
        # CONV2 => POOL2 => NORM2 layers
        model.add(ZeroPadding2D(padding=(2,2)))
        model.add(Conv2D(256, (5, 5), padding='valid', strides=1, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(BatchNormalization())
        # CONV3 => CONV4 => CONV5 => POOL3 layers
        model.add(ZeroPadding2D(padding=(1,1)))
        model.add(Conv2D(384, (3, 3), padding='valid', strides=1, activation='relu'))
        model.add(ZeroPadding2D(padding=(1,1)))
        model.add(Conv2D(384, (3, 3), padding='valid', strides=1, activation='relu'))
        model.add(ZeroPadding2D(padding=(1,1)))
        model.add(Conv2D(256, (3, 3), padding='valid', strides=1, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        # FC6 => FC7 => FC8 layers
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(classes, activation='softmax'))
        # load the weights of a pre-trained model if a path is supplied
        if weightsPath is not None:
            model.load_weights(weightsPath)
        # return the constructed network architecture
        return model
