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
# original_image = imread("Images/input_images/lens.png", 0)
assert original_image is not None, "file could not be read"
# Task 1
# # a) Average Smoothing Filter From Sratch
# updated_image = apply_smoothening_filter(original_image, AVERAGING_FILTER_3x3)
# output_image("t1a.tif", updated_image)


# # b) Gaussian Smoothing Filter
# gaussian_filter_7x7 = generate_gaussian_kernel(size=7, sigma=1, mean=0)
# updated_image = apply_smoothening_filter(original_image, gaussian_filter_7x7)
# output_image("t1b.tif", updated_image)

# # c) Sobel Sharpening Filter
# updated_image = apply_sobel_sharpening_filter(original_image)
# output_image("t1c.tif", updated_image)


# # Task 2
# # Inbuilt CV functions
# # a) Open CV inbuilt averaging function
# averaged_image = blur(original_image, (3, 3))
# output_image("t2a.tif", averaged_image)

# # b) Open CV in built gaussian function
# gaussian_image = GaussianBlur(original_image, (7, 7), 1)
# output_image("t2b.tif", averaged_image)

# c) OpenCV built in function for sobel sharpening filter
# sobel_image = open_cv_sobel(original_image, 3)
# output_image("t2c.tif", sobel_image)


# # Task 3
# # a) The Marr-Hildreth edge detector
# marr_hildreth_image = marr_hildreth_edge_detector(original_image, 3)
# output_image("t3a.tif", marr_hildreth_image)

# # b) The Canny edge detector
# canny_edge_detected_image = canny_edge_detector(original_image)
# output_image("t3b.tif", marr_hildreth_image)
