#!/usr/bin/python
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3,1024) #set horizontal resolution
cap.set(4,768) #set vertical resolution

# square kernel
kernel = np.ones((12,12),np.uint8)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to hsv and find range of colors
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    red_threshold = cv2.inRange(hsv,np.array((0, 80, 80)), np.array((40, 255, 255)))
    blue_threshold = cv2.inRange(hsv,np.array((100, 80, 80)), np.array((140, 255, 255)))
    
    red_threshold_copy = red_threshold.copy()
    blue_threshold_copy = blue_threshold.copy()
    
    # Find contours in the threshold image
    red_contours, red_hierarchy = cv2.findContours(red_threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, blue_hierarchy = cv2.findContours(blue_threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    # Finding contour with maximum area and store it as best_cnt
    red_max_area = 0
    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        if area > red_max_area:
            red_max_area = area
            best_red_cnt = cnt

    blue_max_area = 0
    for cnt in blue_contours:
        area = cv2.contourArea(cnt)
        if area > blue_max_area:
            blue_max_area = area
            best_blue_cnt = cnt

    # finding centroids of best_cnt and draw a circle there
    M = cv2.moments(best_red_cnt)
    cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    cv2.circle(frame,(cx,cy),5,(255,0,0),-1)

    M = cv2.moments(best_blue_cnt)
    cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    cv2.circle(frame,(cx,cy),5,(0,255,0),-1)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #grayscale
    ret,threshold = cv2.threshold(gray,95,255,cv2.THRESH_BINARY) #threshold
    erosion = cv2.erode(threshold,kernel,iterations = 1)

    # Display the resulting frame
    cv2.imshow('red',red_threshold_copy)
    cv2.imshow('blue',blue_threshold_copy)

    cv2.imshow('grayscale',gray)
    cv2.imshow('threshold',threshold)
    cv2.imshow('erosion',erosion)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.imwrite("testgray.png",gray)
cv2.imwrite("testthreshold.png",threshold)
cv2.imwrite("testerosion.png",erosion)
cap.release()
cv2.destroyAllWindows()
