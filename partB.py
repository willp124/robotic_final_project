"""
ref: 
https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
https://www.geeksforgeeks.org/python-opencv-find-center-of-contour/
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # img = cv2.imread('./images/er7-1.jpg')
    # img = cv2.imread('./images/er7-2.jpg')
    img = cv2.imread('./images/er7-3.jpg')
    # img = cv2.imread('./images/er7-4.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(h, w)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    # ret, thresh = cv2.threshold(img, 127, 255, 0)
    # thresh = np.array(thresh,np.uint8)
    contours, hierarchies = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(thresh.shape[:2], dtype='uint8')
    cv2.drawContours(blank, contours, -1, (255, 0, 0), 1)
    # cv2.imwrite("Contours.png", blank)

    cx_list = []
    cy_list = []
    angle_list = []
    for i, c in enumerate(contours):
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # u20= M["m20"] / M["m00"] - cx**2
            # u02= M["m02"] / M["m00"] - cy**2
            # u11= M["m11"] / M["m00"] - cx*cy
            cx_float = round(M['m10']/M['m00'], 4)
            cy_float = round(M['m01']/M['m00'], 4)       
            print(cx, cy)

        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Ignore contours that are too small
        if area < 1000:
            continue
        
        # cv.minAreaRect returns:
        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Retrieve the key parameters of the rotated bounding box
        center = (int(rect[0][0]), int(rect[0][1])) 
        width = float(rect[1][0])
        height = float(rect[1][1])
        angle = float(rect[2])
        print(angle)
            
        if width < height:
            angle = 90 - angle
        else:
            angle = -angle
        # angle = 0.5*math.atan2(2*u11, u20-u02)

        label = "Principal angle: " + str(round(angle,4)) + " degrees"
        if angle != 0: # drop the center of the whole image
            cv2.line(img, (0, int(cy-cx*math.tan(-angle*math.pi/180))), (cx+1000, int(cy+1000*math.tan(-angle*math.pi/180))), (200, 0, 0), 3)
            gradient=math.tan(angle)
            # cv2.line(img, (cx,cy), (cx+300, (cy + int(300*gradient))), (200,0,0), 1)
            # cv2.line(img, (cx,cy), (cx-300, (cy - int(300*gradient))), (200,0,0), 1)
            # cv2.putText(img, label, (center[0]-200, center[1]+100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(img, label, (center[0]-90, center[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            angle_list.append(round(-angle, 4))
            # angle_list.append(round(angle*4068/71, 4))
            if cx != 359 and cy != 239: # drop the center of the whole image
                # cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                # cv2.putText(img, f"center: ({cx_float}, {cy_float})", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                # cv2.putText(img, f"center: ({cx_float}, {cy_float})", (cx - 80, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cx_list.append(cx_float)
                cy_list.append(cy_float)
    # cv2.imwrite("output_er7-1.jpg", img)
    # cv2.imwrite("output_er7-2.jpg", img)
    # cv2.imwrite("output_er7-3.jpg", img)
    # cv2.imwrite("output_er7-4.jpg", img)
    result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # result = img
    k = 0
    output_text = ""
    while k < len(angle_list) :
        output_text = output_text + f"Center {k+1}: (" + str(cx_list[k]) + "," + str(cy_list[k])+f")\n Principal angle {k+1}: " + str(angle_list[k]) + "  degree\n"
        k = k + 1
     
    plt.imshow(result)
    plt.draw()
    plt.subplots_adjust(bottom=0.1)
    # plt.subplots_adjust(bottom=0.35)
    text = plt.figtext(0.5, 0, output_text ,va="baseline", ha="center", fontsize=10)
    # plt.savefig("output_er7-1-1.jpg") 
    # plt.savefig("output_er7-2-1.jpg") 
    plt.savefig("output_er7-3-1.jpg") 
    # plt.savefig("output_er7-4-1.jpg")
    plt.show(block=False)
    first_run = False