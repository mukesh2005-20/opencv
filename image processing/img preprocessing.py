import cv2
import numpy as np

def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or unable to load.")
        return

    # Resize image for display
    max_width = 800
    max_height = 600
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height, 1)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Display resized image
    cv2.imshow("Resized Image", resized_image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    cv2.imshow('Grayscale', gray_resized)

    # Remove noise with Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred_resized = cv2.resize(blurred, new_size, interpolation=cv2.INTER_AREA)
    cv2.imshow('Blurred', blurred_resized)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    edges_resized = cv2.resize(edges, new_size, interpolation=cv2.INTER_AREA)
    cv2.imshow('Canny Edges', edges_resized)

    # Simple binary threshold
    _, binary_thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    binary_thresh_resized = cv2.resize(binary_thresh, new_size, interpolation=cv2.INTER_AREA)
    cv2.imshow('Binary Threshold', binary_thresh_resized)

    # Adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 7, 2
    )
    adaptive_thresh_resized = cv2.resize(adaptive_thresh, new_size, interpolation=cv2.INTER_AREA)
    cv2.imshow('Adaptive Threshold', adaptive_thresh_resized)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(binary_thresh, kernel, iterations=1)
    erosion_resized = cv2.resize(erosion, new_size, interpolation=cv2.INTER_AREA)
    cv2.imshow('Erosion', erosion_resized)

    dilation = cv2.dilate(binary_thresh, kernel, iterations=1)
    dilation_resized = cv2.resize(dilation, new_size, interpolation=cv2.INTER_AREA)
    cv2.imshow('Dilation', dilation_resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_path = "C:/Users/USER/Downloads/temble.jpg"  # Replace with your image path
    preprocess_image(img_path)
