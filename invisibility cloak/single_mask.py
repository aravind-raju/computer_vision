import cv2
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
# Input argument
parser.add_argument("--video", help="Path to input video file. Skip this argument to capture frames from a camera.")

args = parser.parse_args()

print("""Harry :  Hey !! Would you like to try my invisibility cloak ??\nIts awesome !!\nPrepare to get invisible .....................""")

# Creating an VideoCapture object
# This will be used for image acquisition later in the code.
#Its argument can be either the device index or the name of a video file.
#A device index is just the number to specify which camera. Normally one camera will be connected.
#so I simply pass 0 (or -1). You can select the second camera by passing 1 and so on. 
cap = cv2.VideoCapture(args.video if args.video else 0)

# We give some time for the camera to setup
time.sleep(3)
count = 0
background = 0

# Capturing and storing the static background frame
for i in range(60):
  ret, background = cap.read()

#background = np.flip(background,axis=1)

while(cap.isOpened()):
  ret, img = cap.read()
  if not ret:
    break
  count += 1
  #img = np.flip(img, axis=1)
  
  # Converting the color space from BGR to HSV
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  lower_green = np.array([60, 50, 60])
  upper_green = np.array([110, 255, 255])
  mask1 = cv2.inRange(hsv, lower_green, upper_green)

  # Refining the mask corresponding to the detected red color
  mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
  mask1 = cv2.dilate(mask1, np.ones((3,3), np.uint8), iterations=1)
  mask2 = cv2.bitwise_not(mask1)

  # Generating the final output
  res1 = cv2.bitwise_and(background, background, mask=mask1)
  res2 = cv2.bitwise_and(img, img, mask=mask2)
  final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

  cv2.imshow('Magic !!!', final_output)
  k = cv2.waitKey(10)
  if k == 27:
    break