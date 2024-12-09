import cv2 as cv
import os
from ImageHandler import ImageHandler



def load_images(directory):
    """-------------------------------------------------------
    Loads all valid images from a given directory.
    Use: image_handlers = load_images(directory)
    -------------------------------------------------------
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Invalid directory: {directory}")

    image_handlers = []
    for image_name in os.listdir(directory):
        img_path = os.path.join(directory, image_name)
        if os.path.isfile(img_path):
            try:
                image_handlers.append(ImageHandler(directory, image_name))
            except ValueError as e:
                print(e) 
    return image_handlers


# Example usage:
input_dir = "Input/imageset1"
output_dir = "Output"

if __name__ == "__main__":
    for image_handler in load_images(input_dir):
        print(image_handler)
        image_handler.save_current_version(output_dir)
        image_handler.apply_new_function(
            cv.GaussianBlur(image_handler.image, (3, 3), 1)
        )
        image_handler.save_all_versions(output_dir)
