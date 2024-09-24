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

from functions import *

# Q1
original_image = imread("Images/lena.tif")
reduced_image = reduce_image_size(original_image)

# Image Interpolation From Scratch
resized_image_1 = nearnest_neighbour_interpolation_scratch(reduced_image)
output_image("lena_nearest_scratch.tif", resized_image_1)
