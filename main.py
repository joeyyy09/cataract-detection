import cv2
import numpy as np
import os


def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def apply_filtering(image):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    return blurred

def segment_image(image):
    # Adaptive thresholding to separate cataract region
    threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return threshold

def detect_cataract(image):
    # Preprocess the image
    preprocessed_img = preprocess_image(image)

    # Apply filtering
    filtered_img = apply_filtering(preprocessed_img)

    # Segment the image
    segmented_img = segment_image(filtered_img)

    # Calculate mean pixel intensity
    mean_intensity = np.mean(filtered_img)

    # Calculate standard deviation
    std_deviation = np.std(filtered_img)

    # Decide cataract severity based on mean intensity
    if mean_intensity < 100:
        severity = "Healthy"
    elif mean_intensity < 150:
        severity = "Mild"
    elif mean_intensity < 200:
        severity = "Moderate"
    else:
        severity = "Severe"

    return severity

def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            images.append(img)
    return images

# Load eye images
normal_images = load_images('processed_images/test/normal')
cataract_images = load_images('processed_images/test/cataract')

# Detect cataract severity for normal images
print("Normal Images:")
normal_image_paths = os.listdir('processed_images/train/normal')
for filename, image in zip(normal_image_paths, normal_images):
    cataract_severity = detect_cataract(image)
    print("File:", filename, "- Cataract severity:", cataract_severity)

# Detect cataract severity for cataract images
print("\nCataract Images:")
cataract_image_paths = os.listdir('processed_images/train/cataract')
for filename, image in zip(cataract_image_paths, cataract_images):
    cataract_severity = detect_cataract(image)
    print("File:", filename, "- Cataract severity:", cataract_severity)

