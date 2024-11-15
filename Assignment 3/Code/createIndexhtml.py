#Bootstrap

import os
import cv2 as cv

# Constants
INPUT_FOLDER = "Input_Images"
OUTPUT_FOLDER = "Output_Images"
JSFILE = '<script src="Code/createIndexhtml.js"></script>'
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
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"}
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


def generate_html_block(input_image, output_image, index):
    """
    Generates an HTML block for displaying an image set.

    Args:
        input_image (str): Path to the input image.
        output_image (str): Path to the processed output image.
        index (int): Image set index.

    Returns:
        str: HTML content block for the image set.
    """
    return f"""
    <div>
        <div class="container col-lg-8 border rounded-3 mb-5 overflow-hidden p-0">
            <div class="d-flex">
                <img class="img w-50" src="{input_image}">
                <img class="img w-50" src="{output_image}">
            </div>
            <div class="card-body p-4">
                <h5 class="card-title">Image Set {index}</h5>
                <p class="card-text">This project demonstrates the application of OpenCV for iris detection in human eyes. 
                The images below show the original input images on the left and the processed output images with detected irises on the right.</p>
            </div>
        </div>
    </div>
    """


# Get input and output images
input_images = list_image_files(os.path.join(CWD, INPUT_FOLDER))
output_images = list_image_files(os.path.join(CWD, OUTPUT_FOLDER))

# Ensure both lists have equal length
if len(input_images) != len(output_images):
    print("Warning: The number of input and output images does not match.")

# Generate HTML content
html_content += "<div id='images'>"
for index, (input_image_path, output_image_path) in enumerate(
    zip(input_images, output_images), start=1
):
    converted_input = convert_to_jpeg(input_image_path)
    converted_output = convert_to_jpeg(output_image_path)
    if converted_input and converted_output:
        html_content += generate_html_block(converted_input, converted_output, index)
html_content += "</div>"

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
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="Code/styles.css">
</head>
<body class="nunito">
    <div class="container col-lg-10 py-5">
        <h1 class="mb-5">OpenCV Iris Detection Project</h1>
        <p class="lead mb-4">
            This project demonstrates the application of OpenCV for iris detection in human eyes. 
            The images below show the original input images on the left and the processed output 
            images with detected irises on the right.
        </p>
        <h2 class="mt-5 mb-2">Methodology</h2>
        <p class="lead mb-4">
            To achieve the above-mentioned task, the procedure used involves image preprocessing steps, followed
            by Hough Circle Transform.
        </p>
        {html_content}
        <h2 class="mb-4">References</h2>
        <ul>
            <li>Daway, H. G., Kareem, H. H., & Hashim, A. R. (2018). Pupil detection based on color difference and circular Hough transform.</li>
        </ul>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# Save HTML to file
with open("index.html", "w", encoding="utf-8") as file:
    file.write(full_html)
