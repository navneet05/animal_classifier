# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:41:41 2020

@author: Navneet Yadav
"""
#%% import
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os
#%% loading images from local dataset & adding labels
# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images('animals/')))
random.seed(42)
random.shuffle(imagePaths)
# loop over the input images
for imagePath in tqdm(imagePaths):
	# load the image, resize the image to be 32x32 pixels (ignoring
	# aspect ratio), flatten the image into 32x32x3=3072 pixel image
	# into a list, and store the image in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32)).flatten()
	data.append(image)
	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2].split("/")[-1]
	labels.append(label)
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
#%% train-test split
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)
#%% convert the labels from integers to vectors 
"""note-for binary classification use Keras.to_categorical function
as the LabelBinarizer will not return a vector"""
lb = LabelBinarizer()  #present in scikit-learn
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
#%% model
# define the 3072-1024-512-3 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))
# initialize our initial learning rate and epochs to train for
INIT_LR = 0.01
EPOCHS = 80
# compile the model using SGD as our optimizer and categorical cross-entropy loss 
#(you'll want to use binary_crossentropy for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
#%% train the neural network
H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY),epochs=EPOCHS,batch_size=32)
#%% evalution
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
#%% save model and LabelBinarizer
model.save("simple_nn_model.h5")
pickle_out = open("simple_nn_lb.pickle","wb")
pickle.dump(lb, pickle_out)
pickle_out.close()
"""
load pickle
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
load model
reconstructed_model = keras.models.load_model("my_h5_model.h5")
"""