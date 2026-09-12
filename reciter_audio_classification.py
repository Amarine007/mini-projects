"""
Dataset: Quran Recitations for Audio Classification (Kaggle:
mohammedalrajeh/quran-recitations-for-audio-classification), downloaded via
`opendatasets`. Audio clips of Quran reciters labeled by reciter name (12
classes).

Model: A small custom CNN (`ReciterAudioModel`) trained from scratch on
mel-spectrograms computed from each audio clip (3 conv/pool blocks followed
by a fully-connected classifier head).

Tools/Techniques: PyTorch, a custom Dataset (audio loading + spectrogram
extraction via librosa), scikit-learn LabelEncoder, pandas, matplotlib,
torchsummary.
"""

import os
import time

import numpy as np
import pandas as pd
import librosa
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
import matplotlib.pyplot as plt


# Hyperparameters / global configuration
DATASET_URL = "https://www.kaggle.com/datasets/mohammedalrajeh/quran-recitations-for-audio-classification"
DATASET_DIR = "quran-recitations-for-audio-classification"  # local folder created by opendatasets

SAMPLE_RATE = 22050
DURATION = 5           # seconds of audio read per clip
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

IMG_HEIGHT = 128        # spectrogram height after resize (frequency bins)
IMG_WIDTH = 256         # spectrogram width after resize (time steps)

TRAIN_FRAC = 0.7
RANDOM_STATE = 7

LR = 1e-4
BATCH_SIZE = 16
EPOCHS = 25

MODEL_SAVE_PATH = "reciter_audio_model.pth"
label_encoder = LabelEncoder()


class CustomAudioDataset(Dataset):
    def __init__(self, dataframe, device):
        self.dataframe = dataframe
        self.device = device
        self.labels = torch.tensor(label_encoder.transform(dataframe["Class"])).to(device)

        # eagerly compute every spectrogram up front so __getitem__ is just a lookup
        self.audios = [torch.tensor(self.get_spectrogram(path)).float() for path in dataframe["FilePath"]]

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        label = self.labels[idx]
        audio = self.audios[idx].unsqueeze(0).to(self.device)  # add the channel dimension
        return audio, label

    def get_spectrogram(self, file_path):
        """Loads an audio clip and converts it into a fixed-size dB-scale mel-spectrogram."""
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        spec_db = librosa.power_to_db(spec, ref=np.max)  # convert power spectrogram to dB scale

        # pad/truncate to a consistent length, then resize to (IMG_HEIGHT, IMG_WIDTH)
        spec_fixed = librosa.util.fix_length(spec_db, size=DURATION * sr // HOP_LENGTH + 1)
        spec_resized = resize(spec_fixed, (IMG_HEIGHT, IMG_WIDTH), anti_aliasing=True)
        return spec_resized


class ReciterAudioModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.flatten = nn.Flatten()
        # (1, 128, 256) -> after 3x conv/pool blocks -> (64, 16, 32)
        self.linear1 = nn.Linear(64 * 16 * 32, 4096)
        self.linear2 = nn.Linear(4096, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)          # -> (16, 128, 256)
        x = self.pooling(x)        # -> (16, 64, 128)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)          # -> (32, 64, 128)
        x = self.pooling(x)        # -> (32, 32, 64)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)          # -> (64, 32, 64)
        x = self.pooling(x)        # -> (64, 16, 32)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.dropout(x)
        x = self.output(x)

        return x


def download_dataset():
    import opendatasets as od
    od.download(DATASET_URL)
    return DATASET_DIR


def load_and_preprocess_data(dataset_dir):
    data_df = pd.read_csv(os.path.join(dataset_dir, "files_paths.csv"))

    # rewrite the paths shipped in the csv (built for a Colab layout) to point at the local copy
    data_df["FilePath"] = dataset_dir + "/Dataset" + data_df["FilePath"]

    return data_df


def visualize_class_distribution(data_df):
    """Plots a pie chart of how many samples belong to each reciter class."""
    class_counts = data_df["Class"].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(class_counts, labels=class_counts.index, autopct="%1.1f%%")
    plt.title("Class Distribution")
    plt.show()


def build_model(num_classes, device):
    return ReciterAudioModel(num_classes).to(device)


def evaluate(model, loader, dataset_size):
    """Runs inference over a loader with gradients disabled and returns accuracy (%)."""
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            total_correct += (torch.argmax(outputs, axis=1) == labels).sum().item()

    return total_correct / dataset_size * 100


def plot_metrics(train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist):
    """Plots training/validation loss and accuracy curves over epochs."""
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    axs[0].plot(train_loss_hist, label="Training Loss")
    axs[0].plot(val_loss_hist, label="Validation Loss")
    axs[0].set_title("Training and Validation Loss over Epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(train_acc_hist, label="Training Accuracy")
    axs[1].plot(val_acc_hist, label="Validation Accuracy")
    axs[1].set_title("Training and Validation Accuracy over Epochs")
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

    # --- Data investigation ---
    print("Data Shape is:", data_df.shape)
    print(data_df.head())

    print("Classes Distribution is:")
    print(data_df["Class"].value_counts())
    visualize_class_distribution(data_df)

    label_encoder.fit(data_df["Class"])  # fit once on the full label set so train/val/test share the same mapping

    # --- Data splitting: 70% train, 15% validation, 15% test ---
    train_df = data_df.sample(frac=TRAIN_FRAC, random_state=RANDOM_STATE)
    remainder_df = data_df.drop(train_df.index)

    val_df = remainder_df.sample(frac=0.5, random_state=RANDOM_STATE)
    test_df = remainder_df.drop(val_df.index)

    print("Training Shape:", train_df.shape)
    print("Validation Shape:", val_df.shape)
    print("Testing Shape:", test_df.shape)

    train_dataset = CustomAudioDataset(train_df, device)
    val_dataset = CustomAudioDataset(val_df, device)
    test_dataset = CustomAudioDataset(test_df, device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_classes = len(data_df["Class"].unique())
    model = build_model(num_classes, device)
    print(model)

    from torchsummary import summary
    summary(model, (1, IMG_HEIGHT, IMG_WIDTH))

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []


    # --- Epoch runthrough ---
    for epoch in range(EPOCHS):
        start_time = time.time()

        model.train()
        total_loss_train = 0
        total_acc_train = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            total_acc_train += (torch.argmax(outputs, axis=1) == labels).sum().item()

        model.eval()
        total_loss_val = 0
        total_acc_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_loss_val += loss.item()
                total_acc_val += (torch.argmax(outputs, axis=1) == labels).sum().item()

        # average loss per batch (len of the loader), not an arbitrary constant, so it
        # stays meaningful regardless of dataset size or batch size
        train_loss = total_loss_train / len(train_loader)
        val_loss = total_loss_val / len(val_loader)
        train_acc = total_acc_train / len(train_dataset) * 100
        val_acc = total_acc_val / len(val_dataset) * 100

        train_loss_hist.append(round(train_loss, 4))
        val_loss_hist.append(round(val_loss, 4))
        train_acc_hist.append(round(train_acc, 4))
        val_acc_hist.append(round(val_acc, 4))

        epoch_time = time.time() - start_time
        print(f"Epoch: {epoch + 1}/{EPOCHS}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}%, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}% "
              f"-- Time/Epoch: {epoch_time:.2f}s")
        print("=" * 30)

    test_acc = evaluate(model, test_loader, len(test_dataset))
    print(f"Accuracy Score is: {test_acc:.2f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    plot_metrics(train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist)


if __name__ == "__main__":
    main()
