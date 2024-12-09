import cv2 as cv
import os


class ImageHandler:
    """-------------------------------------------------------
    Handles loading, saving, and version management for images.

    Use: handler = ImageHandler(root, image_name)
    -------------------------------------------------------
    Parameters:
        root - path to the directory containing the image (str)
        image_name - name of the image file (str)
    -------------------------------------------------------
    """

    def __init__(self, root, image_name):
        self.name = image_name
        self.history = []
        # Load the image and add it to history
        img_path = os.path.join(root, image_name)
        image = cv.imread(img_path)
        if image is None:
            raise ValueError(f"Invalid image file: {img_path}")
        self.history.append(image)

        self.image = self.history[-1]

    def apply_new_function(self, newImage):
        self.history.append(newImage)

    def save_all_versions(self, root):
        os.makedirs(root, exist_ok=True)
        for i, version in enumerate(self.history):
            output_file = os.path.join(root, f"v_{i}_{self.name}")
            cv.imwrite(output_file, version)

    def save_current_version(self, root):
        os.makedirs(root, exist_ok=True)
        output_file = os.path.join(root, self.name)
        cv.imwrite(output_file, self.image)

    def __str__(self):
        return f"ImageHandler(name={self.name}, versions={len(self.history)})"
