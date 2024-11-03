# # Importing OpenCV package 
# import cv2 

# # Reading the image 
# img = cv2.imread('people.jpg') 

# # Converting image to grayscale 
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# # Loading the required haar-cascade xml classifier file 
# haar_cascade = cv2.CascadeClassifier('myvenv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml') 

# # Applying the face detection method on the grayscale image 
# faces_rect = haar_cascade.detectMultiScale(gray_img, 1.05, 5) 

# # Iterating through rectangles of detected faces 
# for (x, y, w, h) in faces_rect: 
# 	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) 

# cv2.imshow('Detected faces', img) 

# cv2.waitKey(0) 

import numpy as np
import cv2

# Read the image
img = cv2.imread('people.jpg')
img = img.astype(np.uint8)
# Resize the image (optional)
# img = cv2.resize(img, (640, 480))  # adjust the size based on your needs

# Convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization (optional)
gray_img = cv2.equalizeHist(gray_img)

# Load the required Haar cascade xml classifier file
haar_cascade = cv2.CascadeClassifier('myvenv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

# Apply the face detection method on the grayscale image
faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

# Iterate through rectangles of detected faces
for (x, y, w, h) in faces_rect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Detected faces', img)
cv2.waitKey(0)