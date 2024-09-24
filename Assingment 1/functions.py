"""
-------------------------------------------------------
CP467 Assignment 1 Custom Functions Library
-------------------------------------------------------
Author:  Jashandeep Singh
ID:      169018282
Email:   sing8282@mylaurier.ca
__updated__ = "2024-09-24"
-------------------------------------------------------
"""

# IMPORTS
from cv2 import resize, imread, imwrite
from os import path, makedirs


# CONSTANTS
SIZE_FACTOR = 0.5


def nearnest_neighbour_interpolation_scratch(img):
    print(img)
    return img


def output_image(filename, img):
    """
    -------------------------------------------------------
    Saves the provided image to the 'output' directory.
    Creates the directory if it does not exist for cross-platform support.
    Use: output_image(filename, img)
    -------------------------------------------------------
    Parameters:
        filename - the name of the file to save (str)
        img - the image to save
    Returns:
        None
    -------------------------------------------------------
    """
    # Path for Cross Platfrom support
    output_dir = path.join(path.dirname(__file__), "output")
    if not path.exists(output_dir):
        makedirs(output_dir)
    # Path to save the image
    path_ = path.join(output_dir, filename)

    # Save Image using Python's built in function
    imwrite(path_, img)
    return None


def reduce_image_size(img):
    """
    -------------------------------------------------------
    Reduces the size of the input image by a fixed reduction factor.
    The height and width are scaled down proportionally.
    Use: reduced_image = reduce_image_size(img)
    -------------------------------------------------------
    Parameters:
        img - the image to reduce
    Returns:
        reduced_image - the resized image
    -------------------------------------------------------
    """
    # Get Original Size of image
    original_heigth, original_width = img.shape[:2]

    # Scaled Down Height and Widht,Converted to int
    new_height = int(original_heigth * (SIZE_FACTOR))
    new_widht = int(original_width * (SIZE_FACTOR))

    # Reduce the size using Inbuilt openCV Function
    reduced_image = resize(img, (new_height, new_widht))

    return reduced_image
