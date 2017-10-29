import cv2
import numpy as np
import math
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
##import matplotlib.pyplot as plt
font = cv2.FONT_HERSHEY_SIMPLEX

#cap = cv2.VideoCapture(0)
#while(cap.isOpened()):
    # read image
#    ret, img = cap.read()

camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size =(640,480))
time.sleep(0.1)
fgbg = cv2.createBackgroundSubtractorMOG2()

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img = frame.array
    # get hand data from the rectangle sub window on the screen
    cv2.rectangle(img, (300,300), (20,20), (0,255,0),0)
    crop_img = img[20:300, 20:300]
    
#    foreground extraction with grabcut
#    mask = np.zeros(img.shape[:2], np.uint8)
#    bgdModel = np.zeros((1,65), np.float64)
#    fgdModel = np.zeros((1,65), np.float64)
#    rect = (50,50,300,300)
#    cv2.grabCut(img, mask,rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
#    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
#    img = img*mask2[:,:,np.newaxis]
#    plt.imshow(img)
#    plt.colorbar()
#    plt.show()

#    cv2.imshow('image', img)
    
    # MOG2 Background Subtraction
    fgmask = fgbg.apply(crop_img)
#    cv2.imshow('fg', fgmask)
    
    # Apply noise removal with Morphology Closing and OPening
    kernel = np.ones((10,10), np.uint8)
    c1 = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    c2 = cv2.morphologyEx(c1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(c2, cv2.MORPH_CLOSE, kernel)
#    cv2.imshow('c1', c1)
#    cv2.imshow('c2', c2)
    cv2.imshow('closing', closing)
    
    # Find contours of the filtered frame
    _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw Contours
    for cnt in contours:
        color = [222, 222, 222]  # contours color
        cv2.drawContours(crop_img, [cnt], -1, color, 3)

    if contours:

        cnt = contours[0]

        # Find moments of the contour
        moments = cv2.moments(cnt)

        cx = 0
        cy = 0
        # Central mass of first order moments
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
            cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

        center = (cx, cy)
        
        # Draw center mass
        cv2.circle(crop_img, center, 15, [0, 0, 255], 2)

        # find the circle which completely covers the object with minimum area
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(crop_img, center, radius, (0, 0, 0), 3)
        area_of_circle = math.pi * radius * radius

        # drawn bounding rectangle with minimum area, also considers the rotation
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(crop_img, [box], 0, (0, 0, 255), 2)

        # approximate the shape
        cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

        # Find Convex Defects
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)

        fingers = 0    
        # Get defect points and draw them in the original image
        if defects is not None:
            # print('defects shape = ', defects.shape[0])
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv2.line(crop_img, start, end, [0, 255, 0], 3)
                cv2.circle(crop_img, far, 8, [211, 84, 0], -1)
                #  finger count
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                area = cv2.contourArea(cnt)

                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    fingers += 1
                    cv2.circle(crop_img, far, 1, [255, 0, 0], -1)

                if len(cnt) >= 5:
                    (x_centre, y_centre), (minor_axis, major_axis), angle_t = cv2.fitEllipse(cnt)

                letter = ''
                if area_of_circle - area < 5000:
                    #  print('A')
                    letter = 'A'
                elif angle_t > 120:
                    letter = 'U'
                elif area > 120000:
                    letter = 'B'
                elif fingers == 1:
                    if 40 < angle_t < 66:
                        # print('C')
                        letter = 'C'
                    elif 20 < angle_t < 35:
                        letter = 'L'
                    else:
                        letter = 'V'
                elif fingers == 2:
                    if angle_t > 100:
                        letter = 'F'
                    # print('W')
                    else:
                        letter = 'W'
                elif fingers == 3:
                    # print('4')
                    letter = '4'
                elif fingers == 4:
                    # print('Ola!')
                    letter = 'Ola!'
                else:
                    if 169 < angle_t < 180:
                        # print('I')
                        letter = 'I'
                    elif angle_t < 168:
                        # print('J')
                        letter = 'J'

                # Prints the letter and the number of pointed fingers and
                print('Fingers = '+str(fingers)+' | Letter = '+str(letter))
        else:
            # prints msg: no hand detected
            cv2.putText(img, "No hand detected", (45, 450), font, 2, np.random.randint(0, 255, 3).tolist(), 2)

        # Show outputs images
        cv2.imshow('img', img)
        cv2.imshow('mog2', fgmask)
        cv2.imshow('crop_img', crop_img)

# convert to grayscale
#    grey = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
#    cv2.imshow('grey', grey)
    
#   select skin color
#    hsv = cv2.cvtColor(crop_img,cv2.COLOR_BGR2HSV)
#    lower = np.array([0,10,60], dtype = "uint8")
#    upper = np.array([20,150,255], dtype = "uint8")
#    mask = cv2.inRange(hsv,lower,upper)
#    res = cv2.bitwise_and(crop_img, crop_img, mask = mask)
#    cv2.imshow('res', res)

    

    # applying gaussian blur
#    value = (35, 35)
#    blurred = cv2.GaussianBlur(grey, value, 0)

    # thresholdin: Otsu's Binarization method
#    _, thresh1 = cv2.threshold(blurred, 127, 255,
#                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # show thresholded image
#    cv2.imshow('Thresholded', thresh1)

#    # check OpenCV version to avoid unpacking error
#    (version, _, _) = cv2.__version__.split('.')
#    if version == '3':
#        image, contours, hierarchy = cv2.findContours(closing.copy(), \
#               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#    elif version == '2':
#        contours, hierarchy = cv2.findContours(closing.copy(),cv2.RETR_TREE, \
#               cv2.CHAIN_APPROX_NONE)

#    # find contour with max area
#   cnt = max(contours, key = lambda x: cv2.contourArea(x))

#    # create bounding rectangle around the contour (can skip below two lines)
#    x, y, w, h = cv2.boundingRect(cnt)
#    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

#    # finding convex hull
#    hull = cv2.convexHull(cnt)

#    # drawing contours
#    drawing = np.zeros(crop_img.shape,np.uint8)
#    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
#    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    # finding convex hull
#    hull = cv2.convexHull(cnt, returnPoints=False)

    # finding convexity defects
#    defects = cv2.convexityDefects(cnt, hull)
#    count_defects = 0
#    cv2.drawContours(closing, contours, -1, (0, 255, 0), 3)

    # applying Cosine Rule to find angle for all defects (between fingers)
    # with angle > 90 degrees and ignore defects
 #   for i in range(defects.shape[0]):
 #       s,e,f,d = defects[i,0]

 #       start = tuple(cnt[s][0])
 #       end = tuple(cnt[e][0])
 #       far = tuple(cnt[f][0])

        # find length of all sides of triangle
 #       a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
 #       b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
 #       c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # apply cosine rule here
  #      angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignore angles > 90 and highlight rest with red dots
  #      if angle <= 90:
  #          count_defects += 1
  #          cv2.circle(crop_img, far, 1, [0,0,255], -1)
        #dist = cv2.pointPolygonTest(cnt,far,True)

        # draw a line from start to end i.e. the convex points (finger tips)
        # (can skip this part)
  #      cv2.line(crop_img,start, end, [0,255,0], 2)
        #cv2.circle(crop_img,far,5,[0,0,255],-1)

    # define actions required
  #  if count_defects == 1:
  #      cv2.putText(img,"This is 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
  #  elif count_defects == 2:
  #      str = "This is 2"
  #      cv2.putText(img, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
  #  elif count_defects == 3:
  #      cv2.putText(img,"This is 3", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
  #  elif count_defects == 4:
  #      cv2.putText(img,"This is 4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
  #  else:
  #      cv2.putText(img,"Hello World!!!", (50, 50),\
  #                  cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    # show appropriate images in windows
  #  cv2.imshow('Gesture', img)
  #  all_img = np.hstack((drawing, crop_img))
  #  cv2.imshow('Contours', all_img)
    
    rawCapture.truncate(0)
    
    k = cv2.waitKey(10)

    if k == 27:
        break
