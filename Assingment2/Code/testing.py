"""
-------------------------------------------------------
CP467 Assignment 2: Testing
-------------------------------------------------------
Author:  Jashandeep Singh
ID:      169018282
Email:   sing8282@mylaurier.ca
__updated__ = "2024-10-12"
-------------------------------------------------------
"""

# IMPORTS
from functions import *


original_image = imread("Images/input_images/lena.tif", 0)

# Task 1
# a) Average Smoothing Filter From Sratch
updated_image = apply_smoothening_filter(original_image, AVERAGING_FILTER_3x3)
output_image("t1a.tif", updated_image)


# b) Gaussian Smoothing Filter
gaussian_filter_7x7 = generate_gaussian_kernel(size=7, sigma=1, mean=0)
updated_image = apply_smoothening_filter(original_image, gaussian_filter_7x7)
output_image("t1b.tif", updated_image)