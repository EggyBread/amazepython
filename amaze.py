#!/usr/bin/python
import numpy as np
import cv2
from rdp import rdp
import serial
from time import sleep
import math

from Queue import Queue
import Image

# Define some constants
CAM_WIDTH = 1024
CAM_HEIGHT = 768

LOWER_GREEN = (45, 63, 63)
UPPER_GREEN = (75, 255, 255)
LOWER_BLUE = (90, 91, 127)
UPPER_BLUE = (105, 255, 255)

KERNEL_SIZE = 8

ENDPOINT_RADIUS = 16

EPSILON = 0.5

STEPS_PER_PIXEL = 0.25

BAUD_RATE = 115200

def getsteps(pixels, factor):
    return math.floor(pixels * factor)

def iswhite(value):
    if value == (255,255,255): # Remove the alpha channel later as it adds to calculation time
        return True

def getadjacent(n):
    x,y = n
    return [(x-1,y),(x,y-1),(x+1,y),(x,y+1)]

def BFS(start, end, pixels):
    queue = Queue()
    queue.put([start]) # Wrapping the start tuple in a list

    while not queue.empty():
        # open_cv_image = np.array(pixels.convert('RGB')) 
        # Convert RGB to BGR 
        # open_cv_image = open_cv_image[:, :, ::-1].copy()
        # cv2.imshow('solving...',open_cv_image)

        path = queue.get()
        pixel = path[-1]

        if pixel == end:
            return path

        for adjacent in getadjacent(pixel):
            x,y = adjacent
            try:
                if iswhite(pixels[x,y]):
                    pixels[x,y] = (127,127,127) # see note
                    new_path = list(path)
                    new_path.append(adjacent)
                    queue.put(new_path)
            except IndexError:
                pass

    print "Queue has been exhausted. No answer was found."

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cap.set(3,CAM_WIDTH) #set horizontal resolution
cap.set(4,CAM_HEIGHT) #set vertical resolution

# square kernel
kernel = np.ones((KERNEL_SIZE,KERNEL_SIZE),np.uint8)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to hsv and find range of colors
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    green_threshold = cv2.inRange(hsv,np.array(LOWER_GREEN), np.array(UPPER_GREEN))
    blue_threshold = cv2.inRange(hsv,np.array(LOWER_BLUE), np.array(UPPER_BLUE))
    
    green_threshold_copy = green_threshold.copy()
    blue_threshold_copy = blue_threshold.copy()
    
    # Find contours in the threshold image
    green_contours, green_hierarchy = cv2.findContours(green_threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, blue_hierarchy = cv2.findContours(blue_threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    # Finding contour with maximum area and store it as best_cnt
    best_green_cnt = None
    best_blue_cnt = None

    green_max_area = 0
    for cnt in green_contours:
        area = cv2.contourArea(cnt)
        if area > green_max_area:
            green_max_area = area
            best_green_cnt = cnt

    blue_max_area = 0
    for cnt in blue_contours:
        area = cv2.contourArea(cnt)
        if area > blue_max_area:
            blue_max_area = area
            best_blue_cnt = cnt

    # Finding centroids of best_cnt and draw a circle there
    if best_green_cnt is not None:
        M = cv2.moments(best_green_cnt)
        cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        start = (cx,cy)
        # print start

    if best_blue_cnt is not None:
        M = cv2.moments(best_blue_cnt)
        cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        
        end = (cx,cy)
        # print end

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #grayscale
    ret,threshold = cv2.threshold(gray,95,255,cv2.THRESH_BINARY) #threshold
    erosion = cv2.erode(threshold,kernel,iterations = 1)

    try:
        # Draw endpoint markers on raw
        cv2.circle(frame,start,ENDPOINT_RADIUS,(255,0,0),2)
        cv2.circle(frame,end,ENDPOINT_RADIUS,(0,255,0),2)

        # Draw white over endpoints on processed frame
        cv2.circle(erosion,start,ENDPOINT_RADIUS,(255,255,255),-1)
        cv2.circle(erosion,end,ENDPOINT_RADIUS,(255,255,255),-1)
    except NameError:
        pass

    # cv2.imshow('green',green_threshold_copy)
    # cv2.imshow('blue',blue_threshold_copy)

    cv2.imshow('processed',erosion)
    cv2.imshow('raw',frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break;
    elif key == 82:
        print "Up pressed."
    elif key == 84:
        print "Down pressed."

raw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
problem = Image.fromarray(erosion).convert('RGB')

base_img = problem
base_pixels = base_img.load()

path = BFS(start, end, base_pixels)
rdp_path = rdp(path,epsilon=EPSILON)

path_problem = problem
path_problem_pixels = path_problem.load()

path_raw = raw
path_raw_pixels = path_raw.load()

rdp_path_problem = problem.copy()
rdp_path_problem_pixels = rdp_path_problem.load()

rdp_path_raw = raw.copy()
rdp_path_raw_pixels = rdp_path_raw.load()

connected = True
try:
    ser = serial.Serial('/dev/ttyACM0', BAUD_RATE)
except serial.SerialException:
    try:
        ser = serial.Serial('/dev/ttyACM1', BAUD_RATE)
    except serial.SerialException:
        try:
            ser = serial.Serial('/dev/ttyACM2', BAUD_RATE)
        except serial.SerialException:
            connected = False

for index, position in enumerate(path):
    x,y = position
    path_problem_pixels[x,y] = (255,0,0)
    path_raw_pixels[x,y] = (255,0,0)

if connected:
    print "Arduino connected. Sending optimized path now..."
else:
	print "Arduino not connected. Saving optimized solution to images only..."
for index, position in enumerate(rdp_path):
    x,y = position
    if connected:
        ser.write(str(getsteps(x,STEPS_PER_PIXEL)) + '\n')
    sleep(0.005)
    if connected:
        ser.write(str(getsteps(y,STEPS_PER_PIXEL)) + '\n')
    sleep(0.005)
    rdp_path_problem_pixels[x,y] = (255,0,0)
    rdp_path_raw_pixels[x,y] = (255,0,0)

print "Done."

if connected:
    ser.write(str('a'))
    print "Path sent to Arduino. path size is " + str(len(rdp_path))

# cv2.imshow('solution',np.array(path_raw_pixels.getdata()))

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

path_problem.save("solution_processed.png")
path_raw.save("solution_raw.png")

rdp_path_problem.save("optimised_solution_processed.png")
rdp_path_raw.save("optimised_solution_raw.png")

# When everything done, release the capture
cv2.imwrite("testgray.png",gray)
cv2.imwrite("testthreshold.png",threshold)
cv2.imwrite("testerosion.png",erosion)
cap.release()
cv2.destroyAllWindows()
