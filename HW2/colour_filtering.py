import numpy as np
from matplotlib.image import imread
from PIL import Image
import cv2
import imutils

#Color filters: they search pixels within a specific color range, and based on the obtained mask, the new image is constructed

def blue_filter(init_image):
    hsv = cv2.cvtColor(init_image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv,(90, 50, 20), (130, 255, 255) )

    y_img_s, x_img_s, c_img = init_image.shape

    filtered_image = np.zeros((y_img_s, x_img_s, c_img))

    for i in range(y_img_s):
            for j in range(x_img_s):
                if mask[i,j] > 0:
                    filtered_image[i,j,:] = init_image[i,j,:]
                #else:
                #    filtered_image[i,j,:] = 10
                    
    return filtered_image.astype(np.uint8)

def orange_filter(init_image):
    hsv = cv2.cvtColor(init_image, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv,(5, 100, 20), (25, 255, 255) )

    y_img_s, x_img_s, c_img = init_image.shape

    filtered_image = np.zeros((y_img_s, x_img_s, c_img))

    for i in range(y_img_s):
            for j in range(x_img_s):
                if mask[i,j] >0:
                    filtered_image[i,j,:] = init_image[i,j,:]
                #else:
                #    filtered_image[i,j,:] = 10
                    
    return filtered_image.astype(np.uint8)

def red_filter(init_image):
    hsv = cv2.cvtColor(init_image, cv2.COLOR_RGB2HSV)

    mask_1 = cv2.inRange(hsv,(0, 10, 20), (5, 255, 255) )
    mask_2 = cv2.inRange(hsv,(175, 10, 20), (180, 255, 255) )

    y_img_s, x_img_s, c_img = init_image.shape

    filtered_image = np.zeros((y_img_s, x_img_s, c_img))

    for i in range(y_img_s):
            for j in range(x_img_s):
                if mask_1[i,j] > 0 or mask_2[i,j] > 0:
                    filtered_image[i,j,:] = init_image[i,j,:]
                #else:
                #    filtered_image[i,j,:] = 10
                    
    return filtered_image.astype(np.uint8)

#Median Blur; Reduces the unwanted noise from the color-processed images
def median_filter(init_image, filter_size=(3,3)):
    x_filter_size = filter_size[1]
    y_filter_size = filter_size[0]

    y_img_s, x_img_s, c_img = init_image.shape

    h = y_filter_size//2    #1/2 the number of pixels that will be removed after the convolution(height)
    w = x_filter_size//2    #1/2 the number of pixels that will be removed after the convolution(width)

    filtered_image = np.zeros((y_img_s, x_img_s, c_img))

    for c in range(c_img):
        for i in range(h, y_img_s-h):
            for j in range(w, x_img_s-w):
                block = init_image[i-h:i-h+y_filter_size, j-w:j-w+x_filter_size, c]
                
                filtered_image[i][j][c] = np.median(block.flatten())

    return filtered_image.astype(np.uint8)


def contour_finding(init_image, filtered_color):
    #Converting to grayscale
    gray = np.dot(filtered_color[...,:3], [0.2989, 0.5870, 0.1140])

    y_img_s, x_img_s = gray.shape

    #Applying thresholding
    thresh = np.zeros((y_img_s, x_img_s), dtype=np.uint8)

    for i in range(y_img_s):
        for j in range(x_img_s):
            if gray[i][j] > 50:
                thresh[i][j] = 255
 
    #Finding contours and sorting them by area, keeping the 3 biggest ones
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    minAR = 1
    maxAR = 4.9

    final_image = init_image.copy()

    #Drawing a green rectangle around the objects of interest (rectangles or squares)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if ar >= minAR and ar <= maxAR:
            cv2.rectangle(final_image, (x, y), (x + w, y + h), (0,255,0),5)
    
    return final_image.astype(np.uint8)


image = imread('images/Gura_Portitei_Scara_020.jpg')

#BLUE POOL
filtered_blue = blue_filter(image)

#for scales >=040: VERY SLOW; use cv2 instead
filtered_blue = cv2.medianBlur(filtered_blue, 9)
#filtered_blue = median_filter(filtered_blue, filter_size=(9,9))

filtered_blue = contour_finding(image, filtered_color=filtered_blue)
img = Image.fromarray(filtered_blue, 'RGB')
img.save("Demo/020_blue.jpeg")


#ORANGE ROOF
filtered_orange = orange_filter(image)

#for scales >=040: VERY SLOW; use cv2 instead
filtered_orange = cv2.medianBlur(filtered_orange, 9)
#filtered_orange = median_filter(filtered_orange, filter_size=(9,9))

filtered_orange = contour_finding(image, filtered_color=filtered_orange)
img = Image.fromarray(filtered_orange, 'RGB')
img.save("Demo/020_orange.jpeg")

#RED HELIPAD
filtered_red = red_filter(image)

#for scales >=040: VERY SLOW; use cv2 instead
filtered_red = cv2.medianBlur(filtered_red, 13)
#filtered_red = median_filter(filtered_red, filter_size=(13,13))

filtered_red = contour_finding(image, filtered_color=filtered_red)
img = Image.fromarray(filtered_red, 'RGB')
#img.save("040_red.jpeg")
img.save("Demo/020_red.jpeg")