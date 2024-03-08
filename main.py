import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def apply_filtering(image):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    return blurred

def segment_image(image):
    # Threshold the image to separate the cataract region
    _, threshold = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    return threshold

def apply_opening(image):
    # Apply morphological opening to remove noise and fill small gaps
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened

def isolate_circles(image):
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        isolated_circles = np.zeros_like(image)
        for circle in circles[0, :]:
            cv2.circle(isolated_circles, (circle[0], circle[1]), circle[2], (255, 255, 255), -1)
        return isolated_circles
    else:
        return None

def invert_image(image):
    # Invert the image
    inverted = cv2.bitwise_not(image)
    return inverted

def detect_cataract(image):
    # Preprocess the image
    preprocessed_img = preprocess_image(image)

    # Apply filtering
    filtered_img = apply_filtering(preprocessed_img)

    # Segment the image
    segmented_img = segment_image(filtered_img)

    # Apply opening
    opened_img = apply_opening(segmented_img)

    # Isolate circles
    isolated_circles = isolate_circles(opened_img)

    if isolated_circles is not None:
        # Invert the image
        inverted_img = invert_image(isolated_circles)

        # Find contours of the inverted image
        contours, _ = cv2.findContours(inverted_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the maximum area (assuming it's the pupil)
        pupil_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > pupil_area:
                pupil_area = area

        # Calculate the total area of the inverted image
        total_area = inverted_img.shape[0] * inverted_img.shape[1]

        # Calculate the cataract area
        cataract_area = total_area - pupil_area

        # Calculate the degree of cataract
        cataract_degree = (cataract_area / total_area) * 100

        return cataract_degree
    else:
        return 0  # No cataract detected

# Load the eye image
image_path = 'processed_images/test/cataract/image_297.png'
# image_path = 'random_image.jpg'
eye_image = cv2.imread(image_path)

# Resize image (if needed)
# eye_image = cv2.resize(eye_image, (desired_width, desired_height))

# Detect cataract severity
cataract_degree = detect_cataract(eye_image)

# Display result
print("Cataract degree:", cataract_degree)
