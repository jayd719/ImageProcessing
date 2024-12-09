"""
-------------------------------------------------------
CP467 Assignment 3 Custom Functions Library
-------------------------------------------------------
Author:  Jashandeep Singh
ID:      169018282
Email:   sing8282@mylaurier.ca
__updated__ = "2024-10-30"
-------------------------------------------------------
"""

from os import path, makedirs
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Contants
OUTPUT_FOLDER = "Output_images"
EDGE_MAPS_FOLDER = "Edge_maps"

MARKING_COLOR = (0, 225, 0)


def save_image(filename, folder, img):
    """
    -------------------------------------------------------
    Saves the provided image to the specified folder.
    Ensures the directory exists before saving the image.
    Use: save_image(filename, folder, img)
    -------------------------------------------------------
    Parameters:
        filename - the name of the file to save (str)
        folder - the directory in which to save the image (str)
        img - the image to save (ndarray)
    Returns:
        None
    -------------------------------------------------------
    """
    # Path for Cross Platfrom support
    output_dir = path.join(path.dirname(__file__).replace("Code", ""), folder)
    if not path.exists(output_dir):
        makedirs(output_dir)
    # Path to save the image
    path_ = path.join(output_dir, filename)

    # Save Image using Python's built in function
    cv.imwrite(path_, img)
    return None


def hough_circle_transform(img, binary_image, image_name):
    """
    -------------------------------------------------------
    Detects circles in a binary image using the Hough Circle Transform.
    Marks detected circles on the original image and saves edge map.
    Use: hough_circle_transform(img, binary_image, image_name)
    -------------------------------------------------------
    Parameters:
        img - the original image to mark circles on (ndarray)
        binary_image - binary image for edge detection (ndarray)
        image_name - name used for saving output files (str)
    Returns:
        img - the original image with circles marked (ndarray)
    -------------------------------------------------------
    """
    # Apply Canny edge detection
    edge_map = cv.Canny(binary_image, 100, 200, apertureSize=3)
    save_image(f"{image_name}_edge.tif", EDGE_MAPS_FOLDER, edge_map)

    # Apply Hough Circle Transform to detect circles in edge map
    circles = cv.HoughCircles(
        edge_map,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=600,
        param1=200,
        param2=15,
        minRadius=0,
        maxRadius=0,
    )
    # If circles are detected, mark them on the original image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            # Draw the circle on the image
            cv.circle(img, (circle[0], circle[1]), circle[2], MARKING_COLOR, 2)
    return img


def process_image(image_path):
    """
    -------------------------------------------------------
    Processes an image by applying blurring, thresholding, and circle detection.
    Saves the processed output image with detected circles marked.
    Use: process_image(image_path)
    -------------------------------------------------------
    Parameters:
        image_path - the file path of the input image to process (str)
    Returns:
        None
    -------------------------------------------------------
    """
    # Load the input image
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    # Get the image name of the image file for naming output files
    image_name = path.splitext(path.basename(image_path))[0]

    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # apply Guassian Blurr to reduce the noise in image using a 3x3 mask and sigma of 0.
    blurred_image = cv.GaussianBlur(gray_image, (3, 3), 0)

    # Threshold the blurred image to create binary images for pupil and iris
    _, binary_image_outter = cv.threshold(blurred_image, 160, 255, cv.THRESH_BINARY)
    _, binary_image_inner = cv.threshold(blurred_image, 50, 255, cv.THRESH_BINARY)

    # Detect and mark circles in both thresholded images
    img = hough_circle_transform(img, binary_image_outter, image_name)
    img = hough_circle_transform(img, binary_image_inner, image_name)

    # Save the final output image with marked circles
    save_image(f"{image_name}_output.tif", OUTPUT_FOLDER, img)


if __name__ == "__main__":

    process_image("Input_Images/iris1.tif")
    process_image("Input_Images/iris2.tif")
    process_image("Input_Images/iris3.tif")
    process_image("Input_Images/iris4.tif")
    process_image("Input_Images/iris5.tif")


