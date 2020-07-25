import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import pandas as pd


file_name = sys.argv[1]
image_list = sys.argv[2]

data = np.load(file_name)
images = pd.read_csv(image_list)
for i in range(0,data.shape[0]):
    image = cv2.imread(images.ix[i,0])
    loss_image = cv2.resize(data[i,:,:],(image.shape[1],image.shape[0]))
    loss_image = loss_image/np.max(np.max(loss_image))
    #image[:,:,0] = loss_image
    cv2.imshow('image',image)
    cv2.imshow('loss',loss_image)
    cv2.waitKey(0)


