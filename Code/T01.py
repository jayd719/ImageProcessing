"""
-------------------------------------------------------
Assignment 4: Model 1

Freeze all layers of the pre-trained model, modify the last layer to have 
an output size of 10 (to match the number of classes in MNIST), train the 
model on the MNIST training set, and test it on the MNIST test set.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models
import torchvision.transforms as T


# CONSTANTS
MNIST_ROOT = "./MNIST"
BATCH_SIZE = 16
LEARNING_RATE = 0.001
OUTPUT_CLASSES = 10

MODEL1_OUTPUT_PATH = "model1.pt"


# Model Setup
def initialize_model(out_features: int):
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
    # Freeze all layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, out_features),
    )
    return model


# Prepare Datasets
def prepare_datasets(root, transform):
    """
    -------------------------------------------------------
    Prepares training, validation, and test datasets for the
    MNIST dataset, applying specified transformations.
    -------------------------------------------------------
    Parameters:
        root - root directory for MNIST data (str)
        transform - torchvision transform pipeline (T.Compose)
    Returns:
        train_loader - DataLoader for training data (DataLoader)
        val_loader - DataLoader for validation data (DataLoader)
        test_loader - DataLoader for test data (DataLoader)
    -------------------------------------------------------
    """
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    # Split training data into training and validation datasets
    test_size = len(test_ds)
    train_size = len(train_ds) - test_size
    train_ds, validation_ds = random_split(train_ds, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        dataset=validation_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader


def train_model(
    model, train_loader, criterion, optimizer, device, epoch=2, log_interval=10
):
    """
    -------------------------------------------------------
    Trains a given model using the specified DataLoader, loss
    criterion, optimizer, and device.
    -------------------------------------------------------
    Parameters:
        model - model to be trained (torch.nn.Module)
        train_loader - DataLoader for training data (DataLoader)
        criterion - loss function (torch.nn.Module)
        optimizer - optimizer for training (torch.optim.Optimizer)
        device - device for computation (torch.device)
        epoch - current epoch number (int, default=2)
        log_interval - interval (int, default = 10)
    Returns:
        None
    -------------------------------------------------------
    """
    model.train(True)

    total_loss = 0.0
    total_accuracy = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero gradients before backward pass.

        outputs = model(inputs)

        correct_preds = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
        total_accuracy += correct_preds / labels.size(0)
        # Compute loss and backpropagate.
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            avg_loss = total_loss / log_interval
            avg_accuracy = (total_accuracy / log_interval) * 100
            print(
                f"Batch {batch_idx}, Loss: {avg_loss:.3f}, Accuracy: {avg_accuracy:.1f}%"
            )

            total_loss = 0.0
            total_accuracy = 0.0


def evaluate_model(model, val_loader, criterion, device, dataset_type):
    """
    -------------------------------------------------------
    Evaluates a model on a validation or test dataset, providing
    loss and accuracy metrics.
    -------------------------------------------------------
    Parameters:
        model - model to evaluate (torch.nn.Module)
        val_loader - DataLoader for validation or test data (DataLoader)
        device - device for computation (torch.device)
        dataset_type - type of dataset ("Validation" or "Test") (str)
    Returns:
        None
    -------------------------------------------------------
    """
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    # Disable gradient computation for validation.
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # Calculate accuracy for the current batch.
            correct_preds = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            total_accuracy += correct_preds / labels.size(0)

            # Compute loss for the current batch.
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    # Calculate averages.
    avg_loss = total_loss / len(val_loader)
    avg_accuracy = (total_accuracy / len(val_loader)) * 100

    print(
        f"{dataset_type} Loss: {avg_loss:.3f}, {dataset_type} Accuracy: {avg_accuracy:.1f}%"
    )


if __name__ == "__main__":
    print("Task 1")

    # Data Preparation
    transform = T.Compose(
        [
            T.Grayscale(num_output_channels=3),
            T.Resize(256),  # resize images to 256 x 256-> input size for alexnet.
            T.ToTensor(),  # convert to tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader, val_loader, test_loader = prepare_datasets(MNIST_ROOT, transform)
    print("DataLoaders:")
    print(f"- Training size: {len(train_loader.dataset)}")
    print(f"- Validation size: {len(val_loader.dataset)}")
    print(f"- Testing size: {len(test_loader.dataset)}")
    print("-" * 20)

    # Device Selection
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {DEVICE.__str__().upper()}")
    print("-" * 20)

    model = initialize_model(out_features=OUTPUT_CLASSES)
    model.to(device=DEVICE)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4) # As Mentioned AlexNet Paper
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, 3):
        print(f"\nEPOCH {epoch}")
        train_model(model, train_loader, criterion, optimizer, DEVICE, epoch)
        evaluate_model(model, val_loader, criterion, DEVICE, dataset_type="Validation")

    torch.save(model.state_dict(), MODEL1_OUTPUT_PATH)

    print("\nEvaluating on Test Data:")
    evaluate_model(model, test_loader, criterion, DEVICE, "Test")
