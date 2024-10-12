"""
-------------------------------------------------------
CP467 Assignment 2 Custom Functions Library
-------------------------------------------------------
Author:  Jashandeep Singh
ID:      169018282
Email:   sing8282@mylaurier.ca
__updated__ = "2024-10-12"
-------------------------------------------------------
"""

# IMPORTS
from cv2 import resize, imread, imwrite
from os import path, makedirs
from numpy import zeros, array

# FILTERS
AVERAGING_FILTER_3x3 = array(
    [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ]
)

AVERAGING_FILTER_7x7 = array(
    [
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],
    ]
)


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
    output_dir = path.join(
        path.dirname(__file__).replace("Code", ""), "Images/output_images"
    )
    if not path.exists(output_dir):
        makedirs(output_dir)
    # Path to save the image
    path_ = path.join(output_dir, filename)

    # Save Image using Python's built in function
    imwrite(path_, img)
    return None


def pad_image(image, kernel):
    """
    Pads the input image based on the size of the provided kernel.

    Parameters:
    image (ndarray): The input image to be padded.
    kernel (ndarray): The filter/kernel used to determine padding size.

    Returns:
    ndarray: The padded image.
    """
    # Determine the padding size
    pad_height = kernel.shape[0] - 1
    pad_width = kernel.shape[1] - 1

    # Calculate the new padded image dimensions
    padded_height = image.shape[0] + 2 * pad_height
    padded_width = image.shape[1] + 2 * pad_width

    padded_image = zeros((padded_height, padded_width), dtype=image.dtype)

    # Place the original image in the center of the padded image
    padded_image[pad_height:-pad_height, pad_width:-pad_width] = image

    return padded_image


def remove_padding(padded_image, kernel):
    """
    Removes the padding from the padded image based on the size of the kernel.

    Parameters:
    padded_image (ndarray): The padded image.
    kernel (ndarray): The filter/kernel used to determine the padding size.

    Returns:
    ndarray: The image with padding removed.
    """
    # Calculate the padding size based on the kernel
    pad_height = kernel.shape[0] - 1
    pad_width = kernel.shape[1] - 1

    # Return the image with padding removed
    return padded_image[pad_height:-pad_height, pad_width:-pad_width]


def average_smoothing_filter_sractch(img):
    padded_image = pad_image(img, AVERAGING_FILTER_3x3)
    pad_height = AVERAGING_FILTER_3x3.shape[0] - 1
    pad_width = AVERAGING_FILTER_3x3.shape[1] - 1

    for row in range(pad_height, padded_image.shape[0] - pad_height):
        for col in range(pad_width, padded_image.shape[1] - pad_width):
            print(padded_image[row, col])

    return remove_padding(padded_image, AVERAGING_FILTER_3x3)
