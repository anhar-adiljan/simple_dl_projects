from research.cnn.networks.lenet import LeNet
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.utils import np_utils
import pandas as pd
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--file', help='(required) data file name', required=True)
ap.add_argument('-m', '--mode', type=int, default=1, help='(optional) whether or not to train the model (0 for no, 1 for yes)')
ap.add_argument('-s', '--save-model', type=int, default=-1, help='(optional) whether or not model should be saved to disk')
ap.add_argument('-l', '--load-model', type=int, default=-1, help='(optional) whether or not pre-trained model should be loaded')
ap.add_argument('-w', '--weights', type=str, help='(optional) path to weights file')
args = vars(ap.parse_args())

# grab the fashion mnist dataset
print('[INFO] Loading Fashion MNIST...')
dataset = pd.read_csv(args['file'])
labels = dataset['label'].as_matrix()
images = dataset.loc[:, dataset.columns != 'label'].as_matrix()

# scale the data to the range [0, 1.0] and split the data
trainData, testData, trainLabels, testLabels = train_test_split(images / 255.0, labels, test_size=0.33)
trainData = trainData.reshape(trainData.shape[0], 28, 28, 1)
testData  = testData.reshape(testData.shape[0], 28, 28, 1)
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels  = np_utils.to_categorical(testLabels, 10)

# initialize the optimizer and model
print('[INFO] compiling model...')
opt = Adam()
model = LeNet.build(width=28, height=28, depth=1, classes=10, weightsPath=args['weights'] if args['load_model'] > 0 else None)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

if args['mode'] > 0:
    print('[INFO] training...')
    model.fit(trainData, trainLabels, batch_size=128, epochs=5, verbose=1)
print('[INFO] evaluating...')
(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
print('[INFO] accuracy: {:.2f}%'.format(accuracy * 100))

if args['save_model'] > 0:
    print('[INFO] dumping weights to file...')
    model.save_weights('output/fashion-mnist_weights.ckpt', overwrite=True)
