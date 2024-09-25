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
from numpy import zeros, uint8, histogram
from math import floor, ceil

# CONSTANTS
L = 255
C = 255
NUMBER_OF_BITS = 8


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


def interpolation_built_in(img, size_factor, interpolation):
    """
    -------------------------------------------------------
    Resizes an image using a built-in interpolation method provided by OpenCV.
    The interpolation method is specified by the user
    Use: new_image = interpolation_built_in(img, size_factor, interpolation)
    -------------------------------------------------------
    Parameters:
        img - the input image to be resized, represented as a NumPy array (ndarray)
        size_factor - the scaling factor for resizing the image (float)
        interpolation - the interpolation method to use  (int)
    Returns:
        new_image - the resized image using the specified interpolation method (ndarray)
    -------------------------------------------------------
    """
    # Calculation of size for new image
    new_height = img.shape[0] * size_factor
    new_widht = img.shape[1] * size_factor

    # Resize using built in function and return image
    return resize(img, (new_height, new_widht), interpolation=interpolation)


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

    # Numpy array to store new image
    new_image = zeros([new_height, new_widht], dtype=img.dtype)

    # Update the pixels of the new image using nearest-neighbor interpolation
    for row in range(new_height):
        for col in range(new_widht):
            nearest_row = floor(row / size_factor)
            nearest_col = floor(col / size_factor)
            new_image[row, col] = img[nearest_row, nearest_col]

    return new_image


def bilinear_interpolation_from_scratch(img, size_factor):
    """
    -------------------------------------------------------
    Resizes an image using bilinear interpolation.
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

    # Numpy array to store new image
    new_image = zeros([new_height, new_widht], dtype=img.dtype)

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
                q = img[int(x), int(y)]
            elif x_ceil == x_floor:
                q1 = img[int(x), int(y_floor)]
                q2 = img[int(x), int(y_ceil)]
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)

            elif y_ceil == y_floor:
                q1 = img[int(x_floor), int(y)]
                q2 = img[int(x_ceil), int(y)]
                q = (q1 * (x_ceil - x)) + (q2 * (x - x_floor))

            else:
                v1 = img[x_floor, y_floor]
                v2 = img[x_ceil, y_floor]
                v3 = img[x_floor, y_ceil]
                v4 = img[x_ceil, y_ceil]
                q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)
            new_image[row, col] = q
    return new_image


def negate_image(img):
    """
    -------------------------------------------------------
    Negates an image by inverting the pixel values.
    Use: negated_img = negate_image(img)
    -------------------------------------------------------
    Parameters:
        img - the input image to be negated (ndarray)
    Returns:
        negated_img - the negated image (ndarray)
    -------------------------------------------------------
    """
    # create new image
    new_image = zeros([img.shape[0], img.shape[1]], dtype=img.dtype)

    # update new image with inverted values from old image
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            new_image[row, col] = L - 1 - img[row, col]
    return new_image


def power_law_transformation(img, gamma):
    """
    -------------------------------------------------------
    Applies a power-law (gamma) transformation to an image.
    Use: transformed_img = power_law_transformation(img, gamma)
    -------------------------------------------------------
    Parameters:
        img - the input image to be transformed, represented as a NumPy array (ndarray)
        gamma - the gamma correction factor (float > 0)
    Returns:
        transformed_img - the image after applying the power-law transformation (ndarray)
    -------------------------------------------------------
    """
    # create new image
    new_image = zeros([img.shape[0], img.shape[1]], dtype=img.dtype)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            new_image[row, col] = C * ((img[row, col] / C) ** gamma)

    return new_image


def bit_plane_slicing(img):
    """
    -------------------------------------------------------
    Performs bit-plane slicing on a grayscale image.
    Image at Index 0 = MSB
    Use: bit_planes = bit_plane_slicing(img)
    -------------------------------------------------------
    Parameters:
        img - the input grayscale image (single channel) represented as a NumPy array (ndarray)
    Returns:
        bit_planes - a list of binary images representing each bit plane (list of ndarrays)
    -------------------------------------------------------
    """
    bit_planes = [
        zeros([img.shape[0], img.shape[1]], dtype=uint8) for i in range(NUMBER_OF_BITS)
    ]

    # Maximum range of grayscale levels
    GRAYSCALE_RANGE = 2**NUMBER_OF_BITS - 1

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            pixel_val = img[row, col]
            # convert pixel data value into binary
            binary_value = bin(pixel_val)[2:].zfill(NUMBER_OF_BITS)
            for i in range(NUMBER_OF_BITS):
                # update the bit plane as per bit
                bit_planes[i][row][col] = int(binary_value[i]) * GRAYSCALE_RANGE

    return bit_planes


def histogram_equalization(img):
    """
    -------------------------------------------------------
    Performs histogram equalization on a grayscale image to improve contrast.
    Use: equalized_img = histogram_equalization(img)
    -------------------------------------------------------
    Parameters:
        img - the input grayscale image (ndarray)
    Returns:
        equalized_img - image after histogram equalization (ndarray)
    -------------------------------------------------------
    """
    # Numpy array to store new image
    new_image = zeros([img.shape[0], img.shape[1]], dtype=img.dtype)
    max_pixel_value = img.max()

    # Arrays to store in computation values
    hist = zeros([256], dtype=int)
    cum_freq = zeros([256], dtype=int)
    new_pixel = zeros([256], dtype=int)

    # Compute the histogram of the input image
    for row in img:
        for pixel in row:
            hist[pixel] += 1

    cum_freq[0] = hist[0]
    # Compute cumulative frequency
    for i in range(1, 256):
        cum_freq[i] = hist[i] + cum_freq[i - 1]

    # Normalize cumulative frequency to map pixel value
    for i in range(256):
        new_pixel[i] = (cum_freq[i] / [cum_freq[-1]]) * max_pixel_value
        
    # Map old pixel values to the new equalized values
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_image[i, j] = new_pixel[img[i, j]]
    return new_image
