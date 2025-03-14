from PIL import Image
from matplotlib.image import imread
import cv2
import numpy as np
import os
import imutils
import sys

#Step 1: extract the range of colors that represent human skin
def filter_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    #The ranges in which skin color tones are, resulted from manually analyzing the images
    mask = cv2.inRange(hsv,(0, 8, 25), (11, 255, 255) )
    mask_2 = cv2.inRange(hsv,(175, 20, 20), (180, 60, 60))

    masked_image = mask | mask_2

    return masked_image.astype(np.uint8)

#Step 2: Removing small, noisy connected components
def remove_noise(mask_image):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_image, connectivity=8)

    output_image = np.zeros(mask_image.shape, dtype="uint8")

    min_size = 300
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            output_image[labels == i] = 255

    #output_image = cv2.erode(output_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=3)

    return output_image

#Step 3: Dilating and eroding the image
def closing(image, ker_size, n_iter):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ker_size)

    output_image = cv2.dilate(image, kernel, iterations=n_iter)
    output_image = cv2.erode(output_image, kernel, iterations=n_iter)

    return output_image.astype(np.uint8)

#Step 4: Extract the elliptical shapes
def extract_faces(image, n_faces, init_image):
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:n_faces]
    
    output_image = init_image.copy()

    for c in cnts:
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        if ar <= .8:
            ellipse = cv2.fitEllipse(c)
            cv2.ellipse(output_image, ellipse, (0,255,0), 3)

    return output_image.astype(np.uint8)

def save_progress(mask, denoised_image, closed_image, final_image, image_name):
    #Creating all the output folder, if non-existent
    if not os.path.exists("Results"):
        os.mkdir("Results")
    if not os.path.exists("Results/1.masked"):
        os.mkdir("Results/1.masked")
    if not os.path.exists("Results/2.denoised"):
        os.mkdir("Results/2.denoised")
    if not os.path.exists("Results/3.closing"):
        os.mkdir("Results/3.closing")
    if not os.path.exists("Results/4.final"):
        os.mkdir("Results/4.final")


    #Saving every image step in its folder
    img = Image.fromarray(mask, 'L')
    img.save("Results/1.masked/"+image_name)

    img = Image.fromarray(denoised_image, 'L')
    img.save("Results/2.denoised/"+image_name)

    img = Image.fromarray(closed_image, 'L')
    img.save("Results/3.closing/"+image_name)

    img = Image.fromarray(final_image, 'RGB')
    img.save("Results/4.final/"+image_name)


def main():
    images_file = sys.argv[1]
    with open(images_file) as file:
        for line in file:
            image_path = line.split()[0]
            num_faces = int(line.split()[1])
            #For saving output files
            image_name = image_path.split("/")[-1]
        
            image = imread(image_path)
    
            #Step 1
            mask = filter_color(image)
            #Step 2
            denoised_image = remove_noise(mask)
            #Step 3
            closed_image = closing(denoised_image, (5,5), 7)
            #Step 4: Extract the elliptical shapes
            final_image = extract_faces(closed_image, num_faces, init_image = image)

            save_progress(mask, denoised_image, closed_image, final_image, image_name=image_name)

if __name__ == "__main__":
    main()



    

    

    