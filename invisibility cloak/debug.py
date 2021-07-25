import cv2
import numpy as np

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(10)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{0}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([60, 0, 0])
        upper_green = np.array([90, 40, 50])
        mask1 = cv2.inRange(hsv, lower_green, upper_green)
        lower_green = np.array([60, 40, 50])
        upper_green = np.array([90, 255, 255])
        mask2 = cv2.inRange(hsv, lower_green, upper_green)
        img_name = "opencv_frame_{0}{1}.png".format(img_counter, "mask2")
        cv2.imwrite(img_name, mask2)
        print("{} written!".format(img_name))
        mask1 = mask1 + mask2
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
        img_name = "opencv_frame_{0}{1}.png".format(img_counter, "morphologyEx")
        cv2.imwrite(img_name, mask1)
        mask1 = cv2.dilate(mask1, np.ones((3,3), np.uint8), iterations=1)
        img_name = "opencv_frame_{0}{1}.png".format(img_counter, "dilate")
        cv2.imwrite(img_name, mask1)
        mask2 = cv2.bitwise_not(mask1)
        img_name = "opencv_frame_{0}{1}.png".format(img_counter, "bitwise_not")
        cv2.imwrite(img_name, mask2)
  
        img_counter += 1

cam.release()

cv2.destroyAllWindows()