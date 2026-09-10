"""
Animal Faces Classification (Cat / Dog / Wildlife) - CNN in PyTorch
=====================================================================
Trains a CNN on the AFHQ dataset (via kagglehub: andrewmvd/animal-faces) to classify
animal face images as cat, dog, or wildlife.

Techniques: Conv2d/ReLU/MaxPool2d feature extraction, Dropout2d + Dropout regularization,
Adam + cross-entropy loss, train/val/test split with per-epoch metric tracking.

Usage: python animal_face_classification.py
Requirements: pandas, numpy, pillow, kagglehub, scikit-learn, torch, torchvision,
torchsummary, matplotlib.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import kagglehub
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchsummary import summary
import matplotlib.pyplot as plt


# Animal Faces (Cat/Dog/Wildlife) - https://www.kaggle.com/datasets/andrewmvd/animal-faces


DATASET_HANDLE = "andrewmvd/animal-faces"

IMAGE_SIZE = 128
TRAIN_FRAC = 0.7
VAL_FRAC_OF_REMAINDER = 0.5
RANDOM_SEED = 42

LR = 1e-4
BATCH_SIZE = 32
EPOCHS = 10

SAMPLE_GRID_ROWS = 3
SAMPLE_GRID_COLS = 3

DROPOUT_CONV_P = 0.1
DROPOUT_FC_P = 0.3


class CustomImageDataset(Dataset):
    """Loads images from disk lazily, by path."""

    def __init__(self, dataframe, label_encoder, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = torch.tensor(label_encoder.transform(dataframe['label']), dtype=torch.long)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class AnimalFaceModel(nn.Module):
    """3-block CNN feature extractor + fully-connected classifier head."""

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # Dropout2d drops whole channels, which suits conv activations better than plain Dropout.
        self.conv_dropout = nn.Dropout2d(p=DROPOUT_CONV_P)
        self.fc_dropout = nn.Dropout(p=DROPOUT_FC_P)
        # Three 2x2 max-pools halve each spatial dim: IMAGE_SIZE -> IMAGE_SIZE // 8
        pooled_size = IMAGE_SIZE // 8
        self.linear = nn.Linear(in_features=128 * pooled_size * pooled_size, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.pooling(x))

        x = self.conv2(x)
        x = self.relu(self.pooling(x))
        x = self.conv_dropout(x)

        x = self.conv3(x)
        x = self.relu(self.pooling(x))
        x = self.conv_dropout(x)

        x = self.flatten(x)
        x = self.relu(self.linear(x))
        x = self.fc_dropout(x)
        x = self.output(x)  # raw logits; CrossEntropyLoss applies softmax internally
        return x


def download_dataset():
    """Downloads (or reuses the cached copy of) the AFHQ dataset via kagglehub."""
    dataset_path = kagglehub.dataset_download(DATASET_HANDLE)
    print(f"Dataset downloaded to: {dataset_path}")
    return dataset_path


def load_and_preprocess_data(dataset_path):
    """Walks dataset_path/<split>/<label>/<image files> into a (image_path, label) DataFrame."""
    image_path = []
    labels = []

    for i in os.listdir(dataset_path):
        split_path = os.path.join(dataset_path, i)
        if os.path.isdir(split_path):
            for label in os.listdir(split_path):
                label_path = os.path.join(split_path, label)
                if os.path.isdir(label_path):
                    for img_file in os.listdir(label_path):
                        image_path.append(os.path.join(label_path, img_file))
                        labels.append(label)

    data_df = pd.DataFrame(zip(image_path, labels), columns=['image_path', 'label'])
    return data_df


def visualize_samples(data_df, n_rows=SAMPLE_GRID_ROWS, n_columns=SAMPLE_GRID_COLS):
    """Displays a grid of randomly sampled images for a quick sanity check of the dataset."""
    _, axrr = plt.subplots(n_rows, n_columns, figsize=(10, 10))
    for i in range(n_rows):
        for j in range(n_columns):
            image = Image.open(data_df.sample(n=1)['image_path'].iloc[0]).convert("RGB")
            axrr[i, j].imshow(image)
            axrr[i, j].axis('off')
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset_path = os.path.join(download_dataset(), "afhq")
    data_df = load_and_preprocess_data(dataset_path)

    label_encoder = LabelEncoder()
    label_encoder.fit(data_df['label'])

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # TRAIN_FRAC for training; remainder split evenly between val and test.
    train = data_df.sample(frac=TRAIN_FRAC, random_state=RANDOM_SEED)
    test = data_df.drop(train.index)
    val = test.sample(frac=VAL_FRAC_OF_REMAINDER, random_state=RANDOM_SEED)
    test = test.drop(val.index)

    train_dataset = CustomImageDataset(train, label_encoder, transform=transform)
    val_dataset = CustomImageDataset(val, label_encoder, transform=transform)
    test_dataset = CustomImageDataset(test, label_encoder, transform=transform)

    visualize_samples(data_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AnimalFaceModel(num_classes=len(label_encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    print(summary(model, input_size=(3, IMAGE_SIZE, IMAGE_SIZE)))

    total_loss_train_plot = []
    total_acc_train_plot = []
    total_loss_val_plot = []
    total_acc_val_plot = []

    for epoch in range(EPOCHS):
        total_loss_train, correct_train, seen_train = 0, 0, 0
        total_loss_val, correct_val, seen_val = 0, 0, 0

        model.train()  # enables dropout
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            total_loss_train += train_loss.item() * labels.size(0)  # weight by batch size
            train_loss.backward()
            correct_train += (torch.argmax(outputs, axis=1) == labels).sum().item()
            seen_train += labels.size(0)
            optimizer.step()

        model.eval()  # disables dropout
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                total_loss_val += val_loss.item() * labels.size(0)
                correct_val += (torch.argmax(outputs, axis=1) == labels).sum().item()
                seen_val += labels.size(0)

        total_loss_train_plot.append(total_loss_train / seen_train)
        total_acc_train_plot.append(correct_train * 100 / seen_train)
        total_loss_val_plot.append(total_loss_val / seen_val)
        total_acc_val_plot.append(correct_val * 100 / seen_val)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - "
                f"Train Loss: {total_loss_train_plot[-1]:.4f}, Train Acc: {total_acc_train_plot[-1]:.2f}% - "
                f"Val Loss: {total_loss_val_plot[-1]:.4f}, Val Acc: {total_acc_val_plot[-1]:.2f}%")


    # Final evaluation on the held-out test set
    model.eval()
    with torch.no_grad():
        total_test_loss, correct_test, seen_test = 0, 0, 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss = criterion(outputs, labels)
            total_test_loss += test_loss.item() * labels.size(0)
            correct_test += (torch.argmax(outputs, axis=1) == labels).sum().item()
            seen_test += labels.size(0)

        avg_test_loss = total_test_loss / seen_test
        avg_test_acc = correct_test * 100 / seen_test

        print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.2f}%")


    fig, axs = plt.subplots(1, 2, figsize=(15, 10))

    axs[0].plot(total_loss_train_plot, label="Train Loss")
    axs[0].plot(total_loss_val_plot, label="Val Loss")
    axs[0].set_title("Loss over Epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(total_acc_train_plot, label="Train Accuracy")
    axs[1].plot(total_acc_val_plot, label="Val Accuracy")
    axs[1].set_title("Accuracy over Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_ylim(0, 100)
    axs[1].legend()
    plt.show()

    

if __name__ == "__main__":
    main()
