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


def output_image(filename, folder, img):
    """
    -------------------------------------------------------
    Saves the provided image to the 'output' directory.
    Creates the directory if it does not exist for cross-platform support.
    Use: output_image(filename, img)
    -------------------------------------------------------
    Parameters:
        filename - the name of the file to save (str)
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


def hough_circle_transform(img, edge_map, threshold):
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

    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        cv.circle(img, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
    return img


def main(image_name):

    img = cv.imread(f"Input_Images/{image_name}")
    # update the image name for saving the resultant images
    image_name = image_name.split(".")[0]

    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # apply Guassian Blurr to reduce the noise in image using a 3x3 mask and sigma of 3.
    blurred_image = cv.GaussianBlur(gray_image, (3, 3), 0)

    _, binary_image = cv.threshold(blurred_image, 160, 255, cv.THRESH_BINARY)
    # apply canny edge dector to the blurred image
    edge_map = cv.Canny(binary_image, 100, 200, apertureSize=3)
    output_image(f"{image_name}_edge.tif", EDGE_MAPS_FOLDER, edge_map)

    cv.imshow("this", img)
    cv.waitKey(0)


main("iris1.tif")
