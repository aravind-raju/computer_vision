import cv2
import pytesseract

from functions import *

image = cv2.imread('test.jpg')
original = image.copy()
gray = get_grayscale(image)
blur = remove_noise(gray)
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
#cv2.rectangle(image, (x,y), (x + w, y + h), (200, 0, 0), 2)
#cv2.imshow('image with box', image)

img = transformed.copy()
h, w, c = img.shape
boxes = pytesseract.image_to_boxes(img) 
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

#cv2.imshow('img', img)
#cv2.waitKey()
gray = get_grayscale(img)
img = remove_noise(gray)
d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
print(d['text'] )