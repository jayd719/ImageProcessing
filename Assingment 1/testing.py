from functions import *

image_name = "Images/lena.tif"
# image_name = 'Images/cameraman.tif'
img = imread(image_name)

reduced_image = reduce_image_size(img)
print(reduced_image.shape)

output_image('lena_test.tif',reduced_image)