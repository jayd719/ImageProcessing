"""
-------------------------------------------------------
Assignment 4: Task 3

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
from T01 import train_model, initialize_model, evaluate_model
from T02 import prepare_datasets, dataset_paths, preprocess_images_in_directory
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim

BATCH_SIZE = 8
LEARNING_RATE = 0.0001
OUTPUT_CLASSES = 10

# Main entry point
if __name__ == "__main__":
    print("Task 3")
    for path in dataset_paths.values():
        preprocess_images_in_directory(path)

    transform = T.Compose(
        [
            T.Resize(256),
            T.Grayscale(num_output_channels=3),
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

    model = initialize_model(OUTPUT_CLASSES)
    model.to(device=DEVICE)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, 10):
        print(f"\nEPOCH {epoch}")
        train_model(model, train_loader, criterion, optimizer, DEVICE, epoch, 1)
        evaluate_model(model, val_loader, criterion, DEVICE, dataset_type="Validation")

    print("\nEvaluating on Test Data:")
    evaluate_model(model, test_loader, criterion, DEVICE, "Test")
