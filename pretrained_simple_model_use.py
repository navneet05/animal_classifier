# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:39:59 2020

@author: Navneet Yadav
"""
#%% import packages 
from tensorflow.keras.models import load_model
import pickle
import cv2
#%% display image function
def imagepr(img):
    cv2.imshow('image', img) 
    k = cv2.waitKey(0) & 0xFF
  
    # wait for ESC key to exit 
    if k == 27:  
        cv2.destroyAllWindows() 
      
    # wait for 's' key to save and exit 
    elif k == ord('s'):  
        cv2.imwrite('match.png',img) 
        cv2.destroyAllWindows() 
#%% load the input image
image = cv2.imread("images/panda.jpg")
output = image.copy()
image = cv2.resize(image, (32, 32))
# scale the pixel values to [0, 1]
image = image.astype("float") / 255.0
# it is not cnn therefore we flatten the image
image = image.flatten()
image = image.reshape((1, image.shape[0]))
#%% loading model and lable binarizer
model = load_model("output/simple_nn_model.h5")
pickle_in = open("output/simple_nn_lb.pickle","rb")
lb = pickle.load(pickle_in)
#%% prediction
preds = model.predict(image)
#%% find the class label index with the largest corresponding probability
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]
#%% result 
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(0, 0, 255), 2)
# show the output image
imagepr(output)