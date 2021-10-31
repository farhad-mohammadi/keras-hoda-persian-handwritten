import cv2
import numpy as np
from keras.models import load_model

image_path=input('Enter image path : ')

#read image
img = cv2.imread(image_path) 

#Image pyramid
img = cv2.pyrDown(img)

#gray image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#resize and remove border
img = cv2.resize(img, (1000,1000))
img = img[300:700, 300:700]

#remove noise
img = cv2.medianBlur(img,5) 

#thresholding
ret, img =cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

#dilation
kernel = np.ones((5,5),np.uint8)
img =cv2.dilate(img, kernel, iterations = 1)

#erosion
kernel = np.ones((5,5),np.uint8)
img = cv2.erode(img, kernel, iterations = 1)

#invert color
img = 255-img

#show image
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Change the range of pixels from 0-255 to 0-1 
img = img/255

#Resize and reshape the image to the train set 
img = cv2.resize(img, (32,32))
img = img.reshape(1, img.shape[0], img.shape[1], 1)

#load model
model = load_model('model\\model.h5')

#predict 
preds = model.predict(img)
preds=np.argmax(preds, axis=1)
print('Detected digit is : ', int(preds[0]))