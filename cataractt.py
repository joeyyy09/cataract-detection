import cv2
import imutils
import numpy as np
from math import hypot

# Initialize variables to store areas and centroids
pupil_area = 0          
cat_area = 0           

cX_pupil = 0            
cY_pupil = 0            
cX_cat = 0              
cY_cat = 0              

# Function to process the image
def select_image():

    # Path to the image
    path = 'processed_images/test/normal/image_304.png'
    # Load the image
    if len(path) > 0:
        img = cv2.imread(path)
        
        # Resize the image for better processing
        img = imutils.resize(img, width=500)            

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Convert image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply 2D filtering
        kernel = np.ones((5,5),np.float32)/25           
        imgfiltered = cv2.filter2D(gray,-1,kernel)      
        
        # Perform morphological operations
        kernelOp = np.ones((10, 10), np.uint8)          
        kernelCl = np.ones((15, 15), np.uint8)          

        thresh_image = cv2.threshold(imgfiltered,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]      
        morpho = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernelOp)               
        
        # Detect circles using Hough Transform
        circles = cv2.HoughCircles(thresh_image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
          
        img_morpho_copy = morpho.copy()                                 

        circle_values_list = np.uint16(np.around(circles))              
        x, y, r = circle_values_list[0,:][0]
        
        rows, cols = img_morpho_copy.shape                              

        # Remove areas outside the detected circle
        for i in range(cols):                                           
            for j in range(rows):                                       
                if hypot(i-x, j-y) > r:                                 
                    img_morpho_copy[j,i] = 0                            

        imgg_inv = cv2.bitwise_not(img_morpho_copy)                     
        
        # Find contours of pupil
        contours0, hierarchy = cv2.findContours(img_morpho_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   
        cimg_pupil = img.copy()                                                                                 
        for cnt in contours0:                                                   
                cv2.drawContours(cimg_pupil, cnt, -1, (0, 255, 0), 3, 8)        
                pupil_area = cv2.contourArea(cnt)                             

        # Find contours of cataract
        contours0, hierarchy = cv2.findContours(imgg_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)      
        cimg_cat = img.copy()    
        for cnt in contours0:                                                  
                if cv2.contourArea(cnt) < pupil_area:
                    cv2.drawContours(cimg_cat, cnt, -1, (0, 255, 0), 3, 8)      
                    cat_area = cv2.contourArea(cnt)                           
                    cataract_percentage = (cat_area / (pupil_area + cat_area)) * 100        
                    
        cv2.waitKey(0)

        return None

select_image()
