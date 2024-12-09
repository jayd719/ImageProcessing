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
from datetime import datetime

original_image = imread("Images/input_images/lena.tif", 0)
# original_image = imread("Images/input_images/cameraman.tif", 0)
assert original_image is not None, "file could not be read"
# Task 1
# a) Average Smoothing Filter From Sratch

start_time = datetime.now()
updated_image = apply_smoothening_filter(original_image, AVERAGING_FILTER_3x3)
print(f"Task 1A: {datetime.now()-start_time}")
output_image("t1a.tif", updated_image)


# b) Gaussian Smoothing Filter
start_time = datetime.now()
gaussian_filter_7x7 = generate_gaussian_kernel(size=7, sigma=1, mean=0)
updated_image = apply_smoothening_filter(original_image, gaussian_filter_7x7)
print(f"Task 1B: {datetime.now()-start_time}")
output_image("t1b.tif", updated_image)

# c) Sobel Sharpening Filter
start_time = datetime.now()
updated_image = apply_sobel_sharpening_filter(original_image)
print(f"Task 1C: {datetime.now()-start_time}")
output_image("t1c.tif", updated_image)


# Task 2
# Inbuilt CV functions
# a) Open CV inbuilt averaging function
start_time = datetime.now()
averaged_image = blur(original_image, (3, 3))
print(f"Task 2A: {datetime.now()-start_time}")
output_image("t2a.tif", averaged_image)

# b) Open CV in built gaussian function
start_time = datetime.now()
gaussian_image = GaussianBlur(original_image, (7, 7), 1)
print(f"Task 2A: {datetime.now()-start_time}")
output_image("t2b.tif", averaged_image)

# c) OpenCV built in function for sobel sharpening filter
start_time = datetime.now()
sobel_image = open_cv_sobel(original_image, 3)
print(f"Task 2C: {datetime.now()-start_time}")
output_image("t2c.tif", sobel_image)


# Task 3
# a) The Marr-Hildreth edge detector
start_time = datetime.now()
marr_hildreth_image = marr_hildreth_edge_detector(original_image, 3)
print(f"Task 3A: {datetime.now()-start_time}")
output_image("t3a.tif", marr_hildreth_image)

# b) The Canny edge detector
start_time = datetime.now()
canny_edge_detected_image = canny_edge_detector(original_image)
print(f"Task 3B: {datetime.now()-start_time}")
output_image("t3b.tif", marr_hildreth_image)

# Task 4
# Group Adjacent Pixels
start_time = datetime.now()
A4 = group_adjacent_pixels(marr_hildreth_image, 4)
print(f"Task 4A: {datetime.now()-start_time}")
output_image("t4a.tif", A4)

start_time = datetime.now()
B4 = group_adjacent_pixels(canny_edge_detected_image, 4)
print(f"Task 4B: {datetime.now()-start_time}")
output_image("t4b.tif", A4)