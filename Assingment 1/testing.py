"""
-------------------------------------------------------
CP467 Assignment 1: Testing
-------------------------------------------------------
Author:  Jashandeep Singh
ID:      169018282
Email:   sing8282@mylaurier.ca
__updated__ = "2024-09-24"
-------------------------------------------------------
"""

# IMPORTS
from functions import *

# CONSTANTS
SIZE_FACTOR = 2


# Q1 Image Interpolation
original_image = imread("Images/lena.tif")
reduced_image = reduce_image_size(original_image, SIZE_FACTOR)

# Nearest neighbor interpolation implementation from scratch
resized_image_1 = nearnest_neighbour_interpolation_scratch(reduced_image, SIZE_FACTOR)
output_image("lena_nearest_scratch.tif", resized_image_1)

# Nearest neighbor interpolation using OpenCV built-in function
resized_image_2 = nearest_neighbout_interpolation_built_int(reduced_image, SIZE_FACTOR)
output_image("lena_nearest_cv.tif", resized_image_2)

# Bilinear interpolation implementation from scratch
resized_image_3 = bilinear_interpolation_from_scratch(reduced_image,SIZE_FACTOR)
output_image('lena_bilinear_scratch.tif',resized_image_3)

# Bilinear interpolation using OpenCV built-in function
resized_image_4 = 1
