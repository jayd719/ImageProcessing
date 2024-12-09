"""
-------------------------------------------------------
Documentation Generator For Computer Vision Projects : Tailwind
-------------------------------------------------------
Author:  Jashandeep Singh
ID:      169018282
Email:   sing8282@mylaurier.ca
__updated__ = "2024-10-30"
------"""

import os
import cv2 as cv

# Constants
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"
JSFILE = '<script src="https://jayd719.github.io/assets/reports/createIndexhtml.js"></script>'
CSSFILE = '<link rel="stylesheet" href="https://jayd719.github.io/assets/reports/styles.css" />'
CWD = os.path.dirname(__file__).replace("Code", "")

html_content = ""


def list_image_files(directory_path):
    """
    Retrieves all image files in the specified directory and its subdirectories.

    Args:
        directory_path (str): The path of the directory to search.

    Returns:
        list: A list of image file paths found within the directory.
    """
    image_extensions = {".jpg",".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"}
    image_files = []
    parent_folder = directory_path.replace(CWD, "")

    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            if os.path.splitext(file_name)[1].lower() in image_extensions:
                image_path = os.path.join(parent_folder, file_name)
                image_files.append(image_path)
    return image_files


def convert_to_jpeg(image_path):
    """
    Converts the specified image to JPEG format and saves it with a '.jpeg' extension.

    Args:
        image_path (str): The path of the image to convert.

    Returns:
        str: The new JPEG file path.
    """
    try:
        output_file_name = f"{os.path.splitext(image_path)[0]}.jpeg"
        image_path_complete = os.path.join(CWD, image_path)

        img = cv.imread(image_path_complete)
        if img is None:
            raise ValueError(f"Could not read the image at {image_path_complete}")

        output_path = os.path.join(CWD, output_file_name)
        success = cv.imwrite(output_path, img)
        if not success:
            raise IOError(f"Failed to save {output_file_name}")

        print(f"Saved {output_file_name}")
        return output_file_name

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def generate_html_block(images, index):
    """
    Generates an HTML block for displaying an image set.

    Args:
        input_image (str): Path to the input image.
        output_image (str): Path to the processed output image.
        index (int): Image set index.

    Returns:
        str: HTML content block for the image set.
    """
    final_str = ""
    for image in images:
        final_str += f"""
        <figure class="">
            <img class="object-cover" src="{image}" alt="{image}">
            <figcaption class="text-sm">{image}</figcaption>
        </figure>
        """

    return f"""
    <div class="max-w-4xl mx-auto my-6">
        <div class="bg-white shadow-md rounded overflow-hidden p-3">
            <div class="grid grid-cols-2 md:grid-cols-3 justify-items-center">
                {final_str}
            </div>
            <div class="p-4">
                <h5 class="text-lg font-semibold">Image Set {index}</h5>
                <p class="text-gray-600 text-sm">
                    This project demonstrates the application of OpenCV for iris detection in human eyes.
                    The images below show the original input images on the left and the processed output
                    images with detected irises on the right.
                </p>
            </div>
        </div>
    </div>
    """


# Get input and output images
input_images = list_image_files(os.path.join(CWD, INPUT_FOLDER))
output_images = list_image_files(os.path.join(CWD, OUTPUT_FOLDER))


# Generate HTML content
sorted_images = {}
for image in input_images:
    converted_output = convert_to_jpeg(image)
    image_name = image.split("/")[-1].split(".")[0]
    sorted_images.update({image_name: [converted_output]})

for image in output_images:
    converted_output = convert_to_jpeg(image)
    image_name = image.split("/")[-1].split("_")[0]
    sorted_images[image_name].append(converted_output)

i = 1
for images in sorted_images.values():
    html_content += generate_html_block(images, i)
    i += 1


# Add JS file reference
html_content += JSFILE

# Full HTML document
full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenCV Iris Detection Project</title>
    <script src="https://cdn.tailwindcss.com"></script>
    {CSSFILE}
</head>
<body class="bg-gray-100 nunito">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-5xl font-bold mb-6  ">OpenCV Iris Detection Project</h1>
        <p class="text-lg text-gray-700 mb-6">
            This project demonstrates the application of OpenCV for iris detection in human eyes.
            The images below show the original input images on the left and the processed output
            images with detected irises on the right.
        </p>
        <h2 class="text-2xl font-semibold mb-1">Methodology</h2>
        <p class="text-md text-gray-700 mb-1">
            To achieve the above-mentioned task, the procedure used involves image preprocessing steps, followed
            by Hough Circle Transform.
        </p>
        {html_content}
        <h2 class="text-2xl font-semibold mt-8 mb-4">References</h2>
        <ul class="list-disc pl-8 text-gray-700">
            <li>
                Daway, H. G., Kareem, H. H., & Hashim, A. R. (2018). Pupil detection based on color difference 
                and circular Hough transform.
            </li>
        </ul>
    </div>
</body>
</html>
"""

# Save HTML to file
with open("index.html", "w", encoding="utf-8") as file:
    file.write(full_html)
