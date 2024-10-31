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
import cv2 as cv

# Contants
OUTPUT = "Output_images"
EDGE_MAPS = "Edge_maps"


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


def SIFT(img):
    # convert the image into grayscale.
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    threshold1 = 100
    threshold2 = 200
    edge_map = cv.Canny(img, threshold1, threshold2)

    output_image("file.tif", EDGE_MAPS, edge_map)
    # construct shift object
    shift = cv.SIFT_create()
    kp = shift.detect(gray_image, None)
    img = cv.drawKeypoints(gray_image, kp, img)
    return img
