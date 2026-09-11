"""
Dataset: Bean Leaf Lesions Classification (Kaggle: marquis03/bean-leaf-
lesions-classification), downloaded via `opendatasets`. Leaf photos labeled
into 3 classes (healthy vs. two disease categories).

Model: GoogLeNet (Inception v1), pretrained on ImageNet via
torchvision.models, with its final fc layer replaced for 3 classes and all
layers fine-tuned.

Tools/Techniques: PyTorch/torchvision, a custom Dataset (lazy image loading
+ transforms), scikit-learn LabelEncoder, pandas, matplotlib, transfer
learning.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt



# Hyperparameters / global configuration
DATASET_URL = "https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification"
DATASET_DIR = "bean-leaf-lesions-classification"  # local folder created by opendatasets

IMAGE_SIZE = 128      
TRAIN_FRAC = 0.7        
RANDOM_STATE = 7     

LR = 1e-3
BATCH_SIZE = 16
EPOCHS = 15

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # normalization stats
IMAGENET_STD = [0.229, 0.224, 0.225]


MODEL_SAVE_PATH = "googlenet_leaf_model.pth"
label_encoder = LabelEncoder() 


class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform, device):
        self.dataframe = dataframe
        self.transform = transform
        self.device = device
        self.labels = torch.tensor(label_encoder.transform(dataframe["category"])).to(device)

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image).to(self.device)
        return image, label


def download_dataset():
    import opendatasets as od
    od.download(DATASET_URL)
    return DATASET_DIR


def load_and_preprocess_data(dataset_dir):
    train_df = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(dataset_dir, "val.csv"))

    data_df = pd.concat([train_df, val_df], ignore_index=True)
    data_df["image:FILE"] = dataset_dir + "/" + data_df["image:FILE"]

    print("Data shape is:", data_df.shape)
    print(data_df.head())
    print("Class distribution:")
    print(data_df["category"].value_counts())

    return data_df


def visualize_samples(dataset, n_rows=3, n_cols=3):
    """Displays a grid of randomly sampled images"""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    fig, axarr = plt.subplots(n_rows, n_cols)
    for row in range(n_rows):
        for col in range(n_cols):
            image = dataset[np.random.randint(0, len(dataset))][0].cpu()
            image = (image * std + mean).clamp(0, 1)  # undo normalization for display
            axarr[row, col].imshow(image.permute(1, 2, 0).numpy())
            axarr[row, col].axis("off")
    plt.tight_layout()
    plt.show()


def build_model(num_classes, device):
    """Loads pretrained GoogLeNet and swaps its classifier head for this dataset's classes."""
    model = models.googlenet(weights="DEFAULT")
    for param in model.parameters():
        param.requires_grad = True  # fine-tune the whole network, not just the head

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def evaluate(model, loader, dataset_size):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            total_correct += (torch.argmax(outputs, axis=1) == labels).sum().item()

    return total_correct / dataset_size * 100


def plot_metrics(loss_history, acc_history):
    """Plots training loss and accuracy curves over epochs."""
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    axs[0].plot(loss_history, label="Training Loss")
    axs[0].set_title("Training Loss over Epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(acc_history, label="Training Accuracy")
    axs[1].set_title("Training Accuracy over Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_ylim([0, 100])
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_dir = download_dataset()
    data_df = load_and_preprocess_data(dataset_dir)

    # split into train/test sets (TRAIN_FRAC / remainder)
    train_df = data_df.sample(frac=TRAIN_FRAC, random_state=RANDOM_STATE)
    test_df = data_df.drop(train_df.index)

    label_encoder.fit(data_df["category"])  # fit once on the full label set so train/test share the same mapping

    # resize + tensor conversion + ImageNet normalization stats (what GoogLeNet was pretrained on)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = CustomImageDataset(train_df, transform, device)
    test_dataset = CustomImageDataset(test_df, transform, device)

    visualize_samples(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(data_df["category"].unique())
    model = build_model(num_classes, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    loss_history = []
    acc_history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_correct = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (torch.argmax(outputs, axis=1) == labels).sum().item()

        # average loss per batch, not per an arbitrary constant, so it stays meaningful
        # regardless of dataset size or batch size
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / len(train_dataset) * 100

        loss_history.append(round(train_loss, 4))
        acc_history.append(round(train_acc, 4))
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}%")

    test_acc = evaluate(model, test_loader, len(test_dataset))
    print(f"Test Accuracy: {test_acc:.2f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    plot_metrics(loss_history, acc_history)


if __name__ == "__main__":
    main()
