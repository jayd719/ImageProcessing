"""
-------------------------------------------------------
Assignment 4: Task 2

Use the same setup as in Task 1 but test the model on 
your custom test set instead of the MNIST test set.
-------------------------------------------------------
Author:  Jashandeep Singh
ID:      169018282
Email:   sing8282@mylaurier.ca
__updated__ = "2023-02-12"
-------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T
import cv2 as cv
import os
from T01 import initialize_model, MODEL1_OUTPUT_PATH
from T01 import evaluate_model

# CONSTANTS
dataset_paths = {
    "train": "./Training Set",
    "valid": "./Test Set",
    "test": "./Validation Set",
}


BATCH_SIZE = 16
LEARNING_RATE = 0.001
OUTPUT_CLASSES = 10


# Preprocess an image: read in grayscale, resize, and overwrite the original file
def preprocess_image(image_path: str, size: tuple[int, int] = (28, 28)) -> None:
    """
    -------------------------------------------------------
    Reads an image in grayscale, resizes it to a specified
    size, and overwrites the original image.
    -------------------------------------------------------
    Parameters:
        image_path - the path to the image file to preprocess (str)
        size - target dimensions for resizing the image
               (tuple[int, int], default=(256, 256))
    Returns:
        None
    -------------------------------------------------------
    """
    image = cv.imread(image_path)
    if image.shape[1] != 28:
        if image is None:
            print(f"Failed to load image: {image_path}")
            return

        inverted_image = ~image
        resized_image = cv.resize(inverted_image, size)

        cv.imwrite(image_path, resized_image)


# Process all supported images in a directory and its subdirectories
def preprocess_images_in_directory(directory: str):
    """
    -------------------------------------------------------
    Walks through a directory and its subdirectories to find
    and preprocess all supported image files. Processes images
    in-place.
    -------------------------------------------------------
    Parameters:
        directory - path to the root directory containing images
                    to preprocess (str)
    Returns:
        None
    -------------------------------------------------------
    """
    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
    }

    for root, _, files in os.walk(directory, topdown=True):
        for file in files:
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                file_path = os.path.join(root, file)
                preprocess_image(file_path)


def prepare_datasets(dataset_paths: dict, transform: T.Compose):
    """
    -------------------------------------------------------
    Prepares DataLoaders for training, validation, and test datasets
    using custom directory paths and transformations.
    -------------------------------------------------------
    Parameters:
        dataset_paths - a dictionary containing paths to the datasets:
                        {
                            "train": str - path to training dataset,
                            "valid": str - path to validation dataset,
                            "test": str - path to test dataset
                        }
        transform - torchvision transform pipeline to apply to datasets (T.Compose)
    Returns:
        train_loader - DataLoader for training data (DataLoader)
        valid_loader - DataLoader for validation data (DataLoader)
        test_loader - DataLoader for test data (DataLoader)
    -------------------------------------------------------
    """
    # Load datasets using ImageFolder and apply transformations
    train_ds = datasets.ImageFolder(root=dataset_paths["train"], transform=transform)
    valid_ds = datasets.ImageFolder(root=dataset_paths["valid"], transform=transform)
    test_ds = datasets.ImageFolder(root=dataset_paths["test"], transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        dataset=valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    return train_loader, valid_loader, test_loader


# Main entry point
if __name__ == "__main__":
    print("Task 2")

    for path in dataset_paths.values():
        preprocess_images_in_directory(path)

    transform = T.Compose(
        [
            T.Grayscale(num_output_channels=3),
            T.Resize(256),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader, val_loader, test_loader = prepare_datasets(dataset_paths, transform)
    print("DataLoaders:")
    print(f"- Testing size: {len(test_loader.dataset)}")
    print("-" * 20)

    # Device Selection
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {DEVICE.__str__().upper()}")
    print("-" * 20)

    model = initialize_model(OUTPUT_CLASSES)
    model.load_state_dict(torch.load(MODEL1_OUTPUT_PATH, weights_only=True))
    model.to(device=DEVICE)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nEvaluating on Test Data:")
    evaluate_model(model, test_loader, criterion, DEVICE, "Test")
