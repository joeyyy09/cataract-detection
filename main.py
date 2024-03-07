import cv2
import numpy as np

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
    _, threshold = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    return threshold

def detect_cataract(image):
    # Preprocess the image
    preprocessed_img = preprocess_image(image)

    # Apply filtering
    filtered_img = apply_filtering(preprocessed_img)

    # Segment the image
    segmented_img = segment_image(filtered_img)

    # Calculate histogram
    hist = cv2.calcHist([filtered_img], [0], None, [256], [0, 256])

    # Calculate mean pixel intensity
    mean_intensity = np.mean(filtered_img)

    # Decide cataract severity based on mean intensity
    if mean_intensity < 100:
        severity = "Healthy"
    elif mean_intensity < 150:
        severity = "Mild"
    elif mean_intensity < 200:
        severity = "Moderate"
    else:
        severity = "Severe"

    return severity, hist

# Load the eye image
image_path = 'processed_images/test/cataract/image_305.png'
# image_path = 'random_image.jpg'
eye_image = cv2.imread(image_path)

# Resize image (if needed)
# eye_image = cv2.resize(eye_image, (desired_width, desired_height))

# Detect cataract severity
cataract_severity, hist = detect_cataract(eye_image)

# Display result
print("Cataract severity:", cataract_severity)

# Display histogram
import matplotlib.pyplot as plt
plt.plot(hist)
plt.title('Histogram of Filtered Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()
