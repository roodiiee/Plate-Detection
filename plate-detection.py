# Necessary Files imported

import numpy as np
import cv2
from PIL import Image
import pytesseract as tess

tess.pytesseract.tesseract_cmd = 'C://Program Files//Tesseract-OCR//tesseract.exe'


# Clean2_Plate Function to Clean the Plate before feeding it to OCR

def clean2_plate(plate):
   gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

   _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
   if cv2.waitKey(0) & 0xff == ord('q'):
       pass
   num_contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

   if num_contours:
       contour_area = [cv2.contourArea(c) for c in num_contours]
       max_cntr_index = np.argmax(contour_area)

       max_cnt = num_contours[max_cntr_index]
       max_cntArea = contour_area[max_cntr_index]
       x,y,w,h = cv2.boundingRect(max_cnt)

       if not ratioCheck(max_cntArea,w,h):
           return plate,None

       final_img = thresh[y:y+h, x:x+w]
       return final_img,[x,y,w,h]

   else:
       return plate, None


# Function to check whether a valid License Plate is Detected or Not.

def ratioCheck(area, width, height):
   ratio = float(width) / float(height)
   if ratio < 1:
       ratio = 1 / ratio
   if (area < 1063.62 or area > 73862.5) or (ratio < 3 or ratio > 6):
       return False
   return True

def isMaxWhite(plate):
   avg = np.mean(plate)
   if(avg>=115):
       return True
   else:
        return False


def ratio_and_rotation(rect):
   (x, y), (width, height), rect_angle = rect

   if(width>height):
       angle = -rect_angle
   else:
       angle = 90 + rect_angle

   if angle>15:
        return False

   if height == 0 or width == 0:
       return False

   area = height*width
   if not ratioCheck(area,width,height):
       return False
   else:
       return True


# Loading the image

img = cv2.imread("Cars39.png")
print("Image Loaded")
cv2.imshow("input",img)
img_copy = img

# Blurring the image

img2 = cv2.GaussianBlur(img, (3,3), 0)
cv2.imshow("Blurred Image",img2)

# Grayscale the Image

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image",img2)


# Edge detection using Sobel Edge Detection

img2 = cv2.Sobel(img2,cv2.CV_8U,1,0,ksize=3)
cv2.imshow("Edge Detection",img2)

# Binarizing the image using OTSU Threshold

_,img2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("Binarized Image",img2)


# Applying Morphological Operations

# creates rectangular structure
element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(22, 3))
morph_img_threshold = img2.copy()

# Morphology- Dilation is followed by an Erosion- useful in closing small holes inside the foreground objects, or small black point on the object

cv2.morphologyEx(src=img2, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
num_contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)


cv2.drawContours(img_copy, num_contours, -1, (0,255,0), 1)
print("\nContours:")
cv2.imshow("Contours",img_copy)


print(num_contours[1])


cv2.drawContours(morph_img_threshold, num_contours, -1, (0,255,0), 1)
print("Contours Showing all closed boundaries:")
cv2.imshow("morph_img_threshold", morph_img_threshold)


print(len(num_contours))


for i,cnt in enumerate(num_contours):

   min_rect = cv2.minAreaRect(cnt)

   if ratio_and_rotation(min_rect):

       x,y,w,h = cv2.boundingRect(cnt)
       plate_img = img[y:y+h,x:x+w]
       print("Number Plate Identified")
       cv2.imshow("Num plate",plate_img)
       if cv2.waitKey(0) & 0xff == ord('q'):
           pass

       if(isMaxWhite(plate_img)):
           clean_plate, rect = clean2_plate(plate_img)
           if rect:
               fg=0
               x1,y1,w1,h1 = rect
               x,y,w,h = x+x1,y+y1,w1,h1
               plate_im = Image.fromarray(clean_plate)
               text = tess.image_to_string(plate_im, lang='eng')
               print("Number  Detected on Plate Text : ",text)
