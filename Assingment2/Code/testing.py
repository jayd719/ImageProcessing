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


# Q1 Average Smoothing Filter From Sratch
updated_image = average_smoothing_filter_sractch(original_image)
output_image("test_image.png", updated_image)
