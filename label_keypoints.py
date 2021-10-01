import cv2
import numpy as np

import matplotlib.pyplot as plt
import random

def get_cmap(n):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    cmap_fn = plt.cm.get_cmap('hsv', n+1)
    colors = [cmap_fn(i + 1)[:3] for i in range(n)]
    random.shuffle(colors)
    cmap = (np.array(colors) * 255.0).astype(np.uint8)
    return cmap



#This will display all the available mouse click events  
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

#This variable we use to store the pixel location
refPt = []

keypts_colour = get_cmap(24)
print(keypts_colour[0])

num_keypoints = 0

#click event function
def click_event(event, x, y, flags, param):
    global num_keypoints 
    if event == cv2.EVENT_LBUTTONDOWN and num_keypoints <24:
        print(x,",",y)
        refPt.append([x,y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(num_keypoints)
        colour = (int(keypts_colour[num_keypoints][0]), int(keypts_colour[num_keypoints][1]), int(keypts_colour[num_keypoints][2]))
        cv2.circle(img, (x,y), 3, colour, -1)
        cv2.putText(img, strXY, (x,y), font, 0.5, (255,255,0), 1)
        num_keypoints+=1
        if num_keypoints >= 10:
            np.savetxt('training_dataset.txt', refPt)
        cv2.imshow("image", img)

    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue)+", "+str(green)+","+str(red)
        cv2.putText(img, strBGR, (x,y), font, 0.5, (0,255,255), 1)
        cv2.imshow("image", img)


#Here, you need to change the image name and it's path according to your directory
img = cv2.imread("test.png")
cv2.imshow("image", img)

#calling the mouse click event
cv2.setMouseCallback("image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
