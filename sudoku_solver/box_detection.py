import cv2
import numpy as np

from functions import *

image = cv2.imread('test.jpg')
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 3)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

#explanation of findContours: https://stackoverflow.com/questions/64345584/how-to-properly-use-cv2-findcontours-on-opencv-version-4-4-0
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

for c in cnts:
	# arc length - distance between 2 points
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    transformed, ordered_corners = perspective_transform(original, approx)
    break

transformed = cv2.rotate(transformed, cv2.ROTATE_90_COUNTERCLOCKWISE)
x, y, w, h = cv2.boundingRect(ordered_corners)
#roi = image[y:y + h, x:x + w]
cv2.rectangle(image, (x, y), (x + w, y + h), (200, 0, 0), 2)
cv2.imshow('image with box', image)
cv2.imshow('transformed', transformed)
cv2.imwrite('board.png', transformed)
cv2.waitKey()
