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
from numpy import zeros, array, exp, mgrid, square, pi, sum

# FILTERS
AVERAGING_FILTER_3x3 = array(
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
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


def filter_calculation(padded_image, row, col, kernel, filter_sum):
    """
    Calculates the smoothed value for a given pixel using the kernel.

    Parameters:
    padded_image (ndarray): The padded image.
    row (int): The row index of the pixel to apply the filter on.
    col (int): The column index of the pixel to apply the filter on.
    kernel (ndarray): The kernel/filter matrix used for smoothing.

    Returns:
    float: The smoothed value for the specified pixel.
    """
    # Calculate half kernel dimensions
    half_kernel_height = (kernel.shape[0] - 1) // 2
    half_kernel_width = (kernel.shape[1] - 1) // 2

    # Initialize the smoothened value and calculate the kernel sum
    smoothen_value = 0

    # Avoid division by zero in case the kernel sum is zero
    if filter_sum == 0:
        filter_sum = 1  # Fallback to avoid division by zero

    # center of kernel
    center_x = kernel.shape[0] // 2
    center_y = kernel.shape[0] // 2

    # Apply the filter by summing the product of the image pixels and kernel values
    for s in range(-half_kernel_height, half_kernel_height + 1):
        for t in range(-half_kernel_width, half_kernel_width + 1):
            smoothen_value += (
                padded_image[row + s, col + t] * kernel[center_x + s, center_y + t]
            )

    # Return smoothened value
    return smoothen_value / filter_sum


def apply_smoothening_filter(image, kernel):
    """
    Applies the given kernel/filter to the input image.

    Parameters:
    image (ndarray): The input image to be filtered.
    kernel (ndarray): The kernel/filter used to process the image.

    Returns:
    ndarray: The filtered image with padding removed.
    """
    # Pad the image based on the kernel size
    padded_image = pad_image(image, kernel)

    # Calculate padding dimensions
    pad_height = kernel.shape[0] - 1
    pad_width = kernel.shape[1] - 1
    filter_sum = sum(kernel)
    # Apply the kernel to each pixel in the image (excluding padded borders)
    for row in range(pad_height, padded_image.shape[0] - pad_height):
        for col in range(pad_width, padded_image.shape[1] - pad_width):
            # Perform filter calculation at the current position
            padded_image[row, col] = filter_calculation(
                padded_image, row, col, kernel, filter_sum
            )

    # Remove padding and return the filtered image
    return remove_padding(padded_image, kernel)


def generate_gaussian_kernel(size, sigma, mean=0):
    """
    Generates a Gaussian kernel of the specified size and standard deviation (sigma),
    with an optional mean adjustment.

    Parameters:
    size (int): The size of the kernel (must be odd).
    sigma (float): The standard deviation of the Gaussian distribution.
    mean (float, optional): The mean to adjust the grid coordinates. Default is 0.

    Returns:
    ndarray: A normalized 2D Gaussian kernel.
    """
    # Calculate the center of the kernel
    center = size // 2

    # Create a grid of (x, y) coordinates
    x, y = mgrid[-center : center + 1, -center : center + 1]

    # Adjust the grid coordinates by the mean (if specified)
    x = x - mean
    y = y - mean

    # Calculate the Gaussian function
    gaussian_kernel = (1 / (2 * pi * sigma**2)) * exp(
        -(square(x) + square(y)) / (2 * sigma**2)
    )

    # Normalize the kernel
    gaussian_kernel /= sum(gaussian_kernel)

    return gaussian_kernel


