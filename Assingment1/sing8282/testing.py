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
from cv2 import INTER_NEAREST, INTER_LINEAR, INTER_CUBIC

# CONSTANTS
SIZE_FACTOR = 2


# Q1 Image Interpolation
original_image = imread("Images/lena.tif", 0)
reduced_image = reduce_image_size(original_image, SIZE_FACTOR)

# Nearest neighbor interpolation implementation from scratch
resized_image_1 = nearnest_neighbour_interpolation_scratch(reduced_image, SIZE_FACTOR)
output_image("lena_nearest_scratch.tif", resized_image_1)

# Nearest neighbor interpolation using OpenCV built-in function
resized_image_2 = interpolation_built_in(reduced_image, SIZE_FACTOR, INTER_NEAREST)
output_image("lena_nearest_cv.tif", resized_image_2)

# Bilinear interpolation implementation from scratch
resized_image_3 = bilinear_interpolation_from_scratch(reduced_image, SIZE_FACTOR)
output_image("lena_bilinear_scratch.tif", resized_image_3)

# Bilinear interpolation using OpenCV built-in function
resized_image_4 = interpolation_built_in(reduced_image, SIZE_FACTOR, INTER_LINEAR)
output_image("lena_bilinear_cv.tif", resized_image_4)

# Bicubic interpolation using OpenCV built-in function
resized_image_5 = interpolation_built_in(reduced_image, SIZE_FACTOR, INTER_CUBIC)
output_image("lena_bicubic_cv.tif", resized_image_5)


# Q2 Point Operations
cameraman_image = imread("Images/cameraman.tif", 0)

# Find the negative of the image
negative_image = negate_image(cameraman_image)
output_image("cameraman_negative.tif", negative_image)

# Apply power-law transformation on the image
power_image = power_law_transformation(cameraman_image, gamma=0.5)
output_image("cameraman_power.tif", power_image)

# Apply bit-plane slicing on the image
sliced_images = bit_plane_slicing(cameraman_image)
i = 0
# Save each bit plane
# Plane 8 - MSB,Plane 1 - LSB
for image in sliced_images:
    output_image(f"cameraman_b{8-i}.tif", image)
    i += 1

# Q3 Histogram Processing
# Apply histogram equalization on image
einstein_image = imread("Images/einstein.tif", 0)
einstein_equalized = histogram_equalization(einstein_image)
output_image("einstein_equalized.tif", einstein_equalized)

# Apply histogram specification
chest_x_ray1 = imread("Images/chest_x-ray1.jpeg",0)
chest_x_ray2 = imread("Images/chest_x-ray2.jpeg",0)
chest_x_ray3 = histogram_specification(source_image=chest_x_ray2, img=chest_x_ray1)
output_image("chest_x-ray3.jpeg", chest_x_ray3)
