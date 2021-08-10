from functions import *

pathImage = "test.jpg"
heightImg = 450
widthImg = 450

img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
imgThreshold = preProcess(img)

imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS

biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 10)  # DRAW THE BIGGEST CONTOUR
    boxContour = imgBigContour.copy()
    x, y, w, h = cv2.boundingRect(biggest)
    #cv2.rectangle(boxContour, (x, y), (x + w, y + h), (200, 0, 0), 2)
    pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    print(imgWarpColored.shape)
    #print(boxes[0].shape)
    #cv2.imshow("Sample", boxes[1])
    #numbers = getPredection(boxes)
    #print(numbers)
    #imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    #numbers = np.asarray(numbers)
    #posArray = np.where(numbers > 0, 0, 1)
    #print(posArray)

cv2.imshow('transformed', imgBigContour)
cv2.imwrite('cell.png', boxes[13])
cv2.waitKey()