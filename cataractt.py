import cv2  # Importing OpenCV library for image processing
import imutils  # Importing imutils library for convenience functions
import numpy as np  # Importing NumPy library for numerical operations
from math import hypot  # Importing hypot function from math module

# Initialize variables to store areas and centroids
pupil_area = 0          
cat_area = 0           

cX_pupil = 0            
cY_pupil = 0            
cX_cat = 0              
cY_cat = 0              

# Function to process the image
def select_image():
    """
    Processes the eye image to estimate the cataract area and percentage.

    Returns:
        None
    """
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
        
        # Apply 2D filtering to smoothen the image
        kernel = np.ones((5,5), np.float32) / 25  # 5x5 kernel for smoothing           
        imgfiltered = cv2.filter2D(gray, -1, kernel)  # Applying 2D convolution with the kernel

        # Perform morphological operations to enhance features
        kernelOp = np.ones((10, 10), np.uint8)  # Kernel for morphological opening
        kernelCl = np.ones((15, 15), np.uint8)  # Kernel for morphological closing

        # Thresholding to separate objects from background
        thresh_image = cv2.threshold(imgfiltered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # Adaptive thresholding using Otsu's method
        morpho = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernelOp)  # Opening operation to remove noise
        
        # Detect circles using Hough Transform
        circles = cv2.HoughCircles(thresh_image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)  # Detecting circles using Hough Transform
          
        img_morpho_copy = morpho.copy()  # Creating a copy of the morphological image

        circle_values_list = np.uint16(np.around(circles))  # Converting circle parameters to integers
        x, y, r = circle_values_list[0,:][0]  # Extracting circle parameters
        
        rows, cols = img_morpho_copy.shape  # Get the dimensions of the image

        # Remove areas outside the detected circle
        for i in range(cols):  # Looping through columns
            for j in range(rows):  # Looping through rows
                if hypot(i - x, j - y) > r:  # Checking if pixel is outside the circle
                    img_morpho_copy[j, i] = 0  # Set pixel value to 0 (black)

        imgg_inv = cv2.bitwise_not(img_morpho_copy)  # Inverting the image (black to white and vice versa)
        
        # Find contours of pupil
        contours0, hierarchy = cv2.findContours(img_morpho_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Finding contours of objects
        cimg_pupil = img.copy()  # Creating a copy of the original image                                                                                 
        for cnt in contours0:  # Looping through the contours
            cv2.drawContours(cimg_pupil, cnt, -1, (0, 255, 0), 3, 8)  # Drawing contours on the image
            pupil_area = cv2.contourArea(cnt)  # Calculating area of the contour                             

        # Find contours of cataract
        contours0, hierarchy = cv2.findContours(imgg_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # Finding contours of objects in inverted image      
        cimg_cat = img.copy()  # Creating a copy of the original image   
        for cnt in contours0:  # Looping through the contours
            if cv2.contourArea(cnt) < pupil_area:  # Checking if contour area is less than pupil area
                cv2.drawContours(cimg_cat, cnt, -1, (0, 255, 0), 3, 8)  # Drawing contours on the image
                cat_area = cv2.contourArea(cnt)  # Calculating area of the contour                           
                cataract_percentage = (cat_area / (pupil_area + cat_area)) * 100  # Calculating percentage of cataract area
                    
        cv2.waitKey(0)  # Wait for a key press

        return None

select_image()
