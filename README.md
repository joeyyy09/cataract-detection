# Cataract Detection Using Image Processing

## Introduction

This repository contains code for analyzing eye images to detect and classify cataracts using two different methods.

## Method 1: Cataract Severity Detection

### Overview

Method 1 focuses on analyzing eye images to determine the severity of cataracts. It uses image preprocessing techniques followed by segmentation and statistical analysis to assess the severity.

### Steps

1. **Preprocessing**: Convert the image to grayscale to simplify further processing.
2. **Filtering**: Apply Gaussian blur to reduce noise and smooth the image.
3. **Segmentation**: Use adaptive thresholding to separate cataract regions from the rest of the image.
4. **Detection**: Calculate mean pixel intensity and standard deviation to determine cataract severity.
5. **Classification**: Assign severity levels based on mean intensity.

### Usage

- `preprocess_image(image)`: Preprocesses the image by converting it to grayscale.
- `apply_filtering(image)`: Applies Gaussian blur to the preprocessed image.
- `segment_image(image)`: Segments the image using adaptive thresholding.
- `detect_cataract(image)`: Detects the severity of cataracts in the given image.

## Method 2: Cataract Area Estimation

### Overview

Method 2 focuses on estimating the area affected by cataracts in eye images. It utilizes contour detection and morphological operations to isolate the cataract region and calculate its area.

### Steps

1. **Preprocessing**: Resize the image and convert it to grayscale.
2. **Filtering**: Apply 2D filtering to the grayscale image.
3. **Segmentation**: Perform morphological operations and Hough Transform to detect the pupil and cataract regions.
4. **Contour Detection**: Find contours of the pupil and cataract regions.
5. **Area Calculation**: Calculate the area of the pupil and cataract regions.
6. **Percentage Calculation**: Determine the percentage of the cataract area relative to the pupil.

### Usage

- `select_image()`: Processes the eye image to estimate the cataract area and percentage.

## Data

The code assumes the presence of image datasets in the following directory structure:
processed_images/
│
├── test/
│ ├── normal/
│ ├── cataract/
│
└── train/
├── normal/
├── cataract/


Place the eye images in the respective `normal` and `cataract` folders within the `test` and `train` directories.

## Requirements

- OpenCV (`cv2`)
- NumPy

Install the required libraries using `pip install opencv-python numpy`.

## How to Run

1. Ensure the image datasets are properly organized in the specified directory structure.
2. Run the respective method functions (`select_image()` for Method 2 and `detect_cataract(image)` for Method 1) with the appropriate image paths.

## Contributors

- Harshith Mente
- ES Thejas
- P Yushmanth Pali Reddy

## License

This project is licensed under the [MIT License](LICENSE).


