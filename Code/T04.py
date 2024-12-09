"""
-------------------------------------------------------
Assignment 4: Task 4

Freeze all layers of the pre-trained model, modify the last layer to have an output
size of 10, train the model on your custom training set, and test it on your custom test set
-------------------------------------------------------
Author:  Jashandeep Singh
ID:      169018282
Email:   sing8282@mylaurier.ca
__updated__ = "2023-02-12"
-------------------------------------------------------
"""

import torch
from T01 import train_model, evaluate_model
from T02 import prepare_datasets, dataset_paths, preprocess_images_in_directory
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from torchvision import models

BATCH_SIZE = 8
LEARNING_RATE = 0.001
OUTPUT_CLASSES = 10


def initialize_model_modified(output_features: int):
    """
    -------------------------------------------------------
    Initializes a pretrained AlexNet model and modifies the
    classifier to adapt to the specified number of output classes.
    -------------------------------------------------------
    Parameters:
        out_features - number of output classes for the model (int)
    Returns:
        model - modified AlexNet model (torch.nn.Module)
    -------------------------------------------------------
    """
    model = models.alexnet(weights="AlexNet_Weights.DEFAULT")
    # model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
    # Freeze Classification Layers Only
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, output_features),
    )
    return model


# Main entry point
if __name__ == "__main__":
    print("Task 4")
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
    print(f"- Training size: {len(train_loader.dataset)}")
    print(f"- Validation size: {len(val_loader.dataset)}")
    print(f"- Testing size: {len(test_loader.dataset)}")
    print("-" * 20)

    # Device Selection
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {DEVICE.__str__().upper()}")
    print("-" * 20)

    model = initialize_model_modified(OUTPUT_CLASSES)
    model.to(device=DEVICE)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, 5):
        print(f"\nEPOCH {epoch}")
        train_model(model, train_loader, criterion, optimizer, DEVICE, epoch, 1)
        evaluate_model(model, val_loader, criterion, DEVICE, dataset_type="Validation")

    print("\nEvaluating on Test Data:")
    evaluate_model(model, test_loader, criterion, DEVICE, "Test")
