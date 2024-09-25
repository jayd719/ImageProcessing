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
from cv2 import INTER_NEAREST
from os import path, makedirs
from numpy import zeros
from math import floor, ceil


def output_image(filename, img):
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
    output_dir = path.join(path.dirname(__file__), "output")
    if not path.exists(output_dir):
        makedirs(output_dir)
    # Path to save the image
    path_ = path.join(output_dir, filename)

    # Save Image using Python's built in function
    imwrite(path_, img)
    return None


def reduce_image_size(img, size_factor):
    """
    -------------------------------------------------------
    Reduces the size of the input image by a fixed reduction factor.
    The height and width are scaled down proportionally.
    Use: reduced_image = reduce_image_size(img)
    -------------------------------------------------------
    Parameters:
        img     - the image to reduce (ndarray)
        size    - ratio to reduce image by (int)
    Returns:
        reduced_image - the resized image
    -------------------------------------------------------
    """
    # Get Original Size of image
    original_heigth, original_width = img.shape[:2]

    # Scaled Down Height and Widht,Converted to int
    new_height = int(original_heigth / size_factor)
    new_widht = int(original_width / size_factor)

    # Reduce the size using Inbuilt openCV Function
    reduced_image = resize(img, (new_height, new_widht))

    return reduced_image


def nearnest_neighbour_interpolation_scratch(img, size_factor):
    """
    -------------------------------------------------------
    Performs nearest-neighbor interpolation on an image to resize it by a
    specified size factor.
    Use: new_image = nearnest_neighbour_interpolation_scratch(img,size)
    -------------------------------------------------------
    Parameters:
        img     - the input image to be resized (ndarray)
        size    - ratio to scale image by (int)
    Returns:
        new_image - the resized image using nearest-neighbor interpolation (ndarray)
    -------------------------------------------------------
    """
    # Sizes for the new image
    new_height = img.shape[0] * size_factor
    new_widht = img.shape[1] * size_factor

    # shape of orginal Image
    original_shape = img.shape[2]

    # Numpy array to store new image
    new_image = zeros([new_height, new_widht, original_shape], dtype=img.dtype)

    # Update the pixels of the new image using nearest-neighbor interpolation
    for row in range(new_height):
        for col in range(new_widht):
            nearest_row = floor(row / size_factor)
            nearest_col = floor(col / size_factor)
            new_image[row, col] = img[nearest_row, nearest_col]

    return new_image


def nearest_neighbout_interpolation_built_int(img, size_factor):
    """
    -------------------------------------------------------
    Performs nearest-neighbor interpolation on an image to resize it by a
    specified size factor using the built in openCV function
    Use: new_image = nearnest_neighbour_interpolation_scratch(img)
    -------------------------------------------------------
    Parameters:
        img - the input image to be resized (ndarray)
        size    - ratio to scale image by (int)
    Returns:
        new_image - the resized image (ndarray)
    -------------------------------------------------------
    """
    # Calculation of size for new image
    new_height = img.shape[0] * size_factor
    new_widht = img.shape[1] * size_factor

    # Resize using built in function and return image
    return resize(img, (new_height, new_widht), interpolation=INTER_NEAREST)


def bilinear_interpolation_from_scratch(img, size_factor):
    """
    -------------------------------------------------------
    Resizes an image using bilinear interpolation. For each pixel in the new image,
    the algorithm computes a weighted average of the four nearest pixels in the
    original image.
    Use: new_image = bilinear_interpolation_from_scratch(img, size_factor)
    -------------------------------------------------------
    Parameters:
        img - the input image to be resized, represented as a NumPy array (ndarray)
        size_factor - the scaling factor for resizing the image (float)
    Returns:
        new_image - the resized image using bilinear interpolation (ndarray)
    -------------------------------------------------------
    """
    # Sizes for the new image
    new_height = img.shape[0] * size_factor
    new_widht = img.shape[1] * size_factor

    # shape of orginal Image
    original_shape = img.shape[2]

    # Numpy array to store new image
    new_image = zeros([new_height, new_widht, original_shape], dtype=img.dtype)

    for row in range(new_height):
        for col in range(new_widht):

            # map the cordinates back to image
            x = row / size_factor
            y = col / size_factor

            # calculate the locations for four neighbouring pixels

            x_floor = floor(x)
            y_floor = floor(y)

            x_ceil = min(img.shape[0] - 1, ceil(x))
            y_ceil = min(img.shape[1] - 1, ceil(y))

            if (x_floor == x_ceil) and (y_floor == y_ceil):
                q = img[int(x), int(y), :]
            elif x_ceil == x_floor:
                q1 = img[int(x), int(y_floor), :]
                q2 = img[int(x), int(y_ceil), :]
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)

            elif y_ceil == y_floor:
                q1 = img[int(x_floor), int(y), :]
                q2 = img[int(x_ceil), int(y), :]
                q = (q1 * (x_ceil - x)) + (q2 * (x - x_floor))

            else:
                v1 = img[x_floor, y_floor, :]
                v2 = img[x_ceil, y_floor, :]
                v3 = img[x_floor, y_ceil, :]
                v4 = img[x_ceil, y_ceil, :]
                q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)
            new_image[row, col] = q
    return new_image
