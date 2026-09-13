"""
ECG Heartbeat Classification (MIT-BIH, 5 classes) - 1D CNN / LSTM in PyTorch
============================================================================
Dataset: ECG Heartbeat Categorization (Kaggle: shayanfazeli/heartbeat), downloaded via
`kagglehub`. Single heartbeats segmented from the MIT-BIH Arrhythmia recordings, each a
187-sample 1D signal (already scaled to [0, 1] and zero-padded) labeled into 5 AAMI classes:
Normal, Supraventricular, Ventricular, Fusion, Unknown.

Model: Either a 1D CNN (`ECGConvModel`, 3 Conv1d blocks + fully-connected head) or a
bidirectional LSTM (`ECGLSTMModel`, over 11-sample patches, max-pooled over time ->
linear head), picked by MODEL_TYPE. Both read the raw waveform as a sequence - no
spectrogram / image conversion, unlike the reciter audio script.

Tools/Techniques: PyTorch, Conv1d / LSTM sequence modeling, class-weighted CrossEntropyLoss
for a heavily imbalanced dataset, stratified train/val/test split, per-class
precision/recall + macro F1 + confusion matrix (accuracy alone is misleading here).

Usage: python ecg_heartbeat_classification.py
Requirements: pandas, numpy, kagglehub, scikit-learn, torch, matplotlib.
"""

import os
import time

import numpy as np
import pandas as pd
import kagglehub
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Hyperparameters / global configuration
DATASET_HANDLE = "shayanfazeli/heartbeat"
DATA_FILES = ["mitbih_train.csv", "mitbih_test.csv"]  # merged, then re-split 70/15/15 below

# label column holds 0-4 already, so no LabelEncoder is needed - just names for display
CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]
SIGNAL_LENGTH = 187

MODEL_TYPE = "cnn"      # "cnn" or "lstm"
USE_CLASS_WEIGHTS = True

TRAIN_FRAC = 0.7
RANDOM_STATE = 7

LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 15

LSTM_PATCH_SIZE = 11    # samples per LSTM time step; must divide SIGNAL_LENGTH (187 = 17 * 11)
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
DROPOUT_P = 0.3

MODEL_SAVE_PATH = "ecg_model.pth"


class ECGDataset(Dataset):
    def __init__(self, signals, labels, device):
        # (N, 187) -> (N, 1, 187): one input channel, like a mono audio clip
        self.X = torch.tensor(signals, dtype=torch.float32).unsqueeze(1).to(device)
        self.y = torch.tensor(labels, dtype=torch.long).to(device)  # CrossEntropyLoss wants long targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ECGConvModel(nn.Module):
    """3-block 1D CNN feature extractor + fully-connected classifier head."""

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),     # -> (32, 187)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),                                # -> (32, 93)

            nn.Conv1d(32, 64, kernel_size=5, padding=2),    # -> (64, 93)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),                                # -> (64, 46)

            nn.Conv1d(64, 128, kernel_size=3, padding=1),   # -> (128, 46)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),                                # -> (128, 23)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (SIGNAL_LENGTH // 8), 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_P),
            nn.Linear(128, num_classes),  # raw logits; CrossEntropyLoss applies softmax internally
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ECGLSTMModel(nn.Module):
    """Bidirectional stacked LSTM over patches of the waveform, max-pooled over time."""

    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=LSTM_PATCH_SIZE, hidden_size=LSTM_HIDDEN_SIZE,
                            num_layers=LSTM_NUM_LAYERS, batch_first=True, dropout=DROPOUT_P,
                            bidirectional=True)
        self.dropout = nn.Dropout(DROPOUT_P)
        self.output = nn.Linear(2 * LSTM_HIDDEN_SIZE, num_classes)  # 2x: forward + backward directions

    def forward(self, x):
        # (batch, 1, 187) -> (batch, 17, 11): 17 time steps of 11 consecutive samples each.
        # Feeding 187 one-sample steps is ~11x slower and much harder for the LSTM to learn.
        x = x.reshape(x.size(0), SIGNAL_LENGTH // LSTM_PATCH_SIZE, LSTM_PATCH_SIZE)
        outputs, _ = self.lstm(x)           # (batch, 17, 2 * hidden)
        # max-pool over every time step rather than taking the last hidden state: beats are
        # zero-padded at the end, so the last step mostly "remembers" padding, not the beat
        x, _ = outputs.max(dim=1)
        x = self.dropout(x)
        return self.output(x)


def download_dataset():
    """Downloads (or reuses the cached copy of) the heartbeat dataset via kagglehub."""
    dataset_path = kagglehub.dataset_download(DATASET_HANDLE)
    print(f"Dataset downloaded to: {dataset_path}")
    return dataset_path


def load_and_preprocess_data(dataset_path):
    """Reads the headerless MIT-BIH csvs: 187 signal columns followed by one label column."""
    data_df = pd.concat(
        [pd.read_csv(os.path.join(dataset_path, f), header=None) for f in DATA_FILES],
        ignore_index=True,
    )

    X = data_df.iloc[:, :-1].values
    y = data_df.iloc[:, -1].astype(int).values  # labels are stored as floats (0.0 - 4.0)
    return X, y


def visualize_class_distribution(y):
    """Bar chart of beats per class - makes the imbalance obvious."""
    counts = np.bincount(y, minlength=len(CLASS_NAMES))

    plt.figure(figsize=(8, 5))
    plt.bar(CLASS_NAMES, counts)
    plt.title("Class Distribution")
    plt.ylabel("Number of beats")
    plt.show()


def visualize_samples(X, y):
    """Plots one randomly chosen heartbeat per class for a quick sanity check."""
    rng = np.random.default_rng(RANDOM_STATE)
    _, axs = plt.subplots(1, len(CLASS_NAMES), figsize=(20, 4), sharey=True)

    for label, ax in enumerate(axs):
        idx = rng.choice(np.where(y == label)[0])
        ax.plot(X[idx])
        ax.set_title(CLASS_NAMES[label])
        ax.set_xlabel("Sample")
    axs[0].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


def build_model(num_classes, device):
    if MODEL_TYPE == "cnn":
        model = ECGConvModel(num_classes)
    elif MODEL_TYPE == "lstm":
        model = ECGLSTMModel(num_classes)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE!r} (expected 'cnn' or 'lstm')")
    return model.to(device)


def predict(model, loader):
    """Runs inference over a loader with gradients disabled; returns (true labels, predictions)."""
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            all_labels.append(labels.cpu())
            all_preds.append(torch.argmax(outputs, axis=1).cpu())

    return torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()


def evaluate(model, loader):
    """Test-set report: accuracy, per-class precision/recall/F1, and a confusion matrix."""
    y_true, y_pred = predict(model, loader)

    test_acc = (y_true == y_pred).mean() * 100
    print(f"Accuracy Score is: {test_acc:.2f}%")
    print(f"Macro F1 is: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

    # normalized per true class, so the rare classes are as readable as "Normal"
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=CLASS_NAMES,
                                            normalize="true", values_format=".2f", cmap="Blues")
    plt.title("Test Confusion Matrix (row-normalized)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return test_acc


def plot_metrics(train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, val_f1_hist):
    """Plots training/validation loss, accuracy, and validation macro F1 over epochs."""
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

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

    axs[2].plot(val_f1_hist, label="Validation Macro F1", color="tab:green")
    axs[2].set_title("Validation Macro F1 over Epochs")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("Macro F1")
    axs[2].set_ylim([0, 1])
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path = download_dataset()
    X, y = load_and_preprocess_data(dataset_path)

    # --- Data investigation ---
    print("Signals Shape is:", X.shape)
    print("Classes Distribution is:")
    for name, count in zip(CLASS_NAMES, np.bincount(y)):
        print(f"  {name:<17} {count:>6}  ({count / len(y) * 100:.1f}%)")
    visualize_class_distribution(y)
    visualize_samples(X, y)

    # --- Data splitting: 70% train, 15% validation, 15% test ---
    # stratified, so the rare classes (Fusion is < 1%) show up in every split in the same proportion
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=TRAIN_FRAC, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE)

    print("Training Shape:", X_train.shape)
    print("Validation Shape:", X_val.shape)
    print("Testing Shape:", X_test.shape)

    train_dataset = ECGDataset(X_train, y_train, device)
    val_dataset = ECGDataset(X_val, y_val, device)
    test_dataset = ECGDataset(X_test, y_test, device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(CLASS_NAMES)
    model = build_model(num_classes, device)
    print(model)
    # torchsummary can't handle the LSTM's tuple output, so just count parameters
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # "balanced" weights = n_samples / (n_classes * class_count), computed on train only: a
    # misclassified Fusion beat costs far more than a misclassified Normal one, so the model
    # can't score well just by predicting "Normal" everywhere
    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
        print("Class weights:", {name: round(float(w), 3) for name, w in zip(CLASS_NAMES, class_weights)})
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    val_f1_hist = []


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
        val_labels, val_preds = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_loss_val += loss.item()
                val_labels.append(labels.cpu())
                val_preds.append(torch.argmax(outputs, axis=1).cpu())

        val_labels = torch.cat(val_labels).numpy()
        val_preds = torch.cat(val_preds).numpy()

        # average loss per batch (len of the loader), not an arbitrary constant, so it
        # stays meaningful regardless of dataset size or batch size
        train_loss = total_loss_train / len(train_loader)
        val_loss = total_loss_val / len(val_loader)
        train_acc = total_acc_train / len(train_dataset) * 100
        val_acc = (val_labels == val_preds).mean() * 100
        # macro F1 weighs every class equally - the number to watch on imbalanced data
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        train_loss_hist.append(round(train_loss, 4))
        val_loss_hist.append(round(val_loss, 4))
        train_acc_hist.append(round(train_acc, 4))
        val_acc_hist.append(round(val_acc, 4))
        val_f1_hist.append(round(val_f1, 4))

        epoch_time = time.time() - start_time
        print(f"Epoch: {epoch + 1}/{EPOCHS}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}%, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}%, "
              f"Validation Macro F1: {val_f1:.4f} "
              f"-- Time/Epoch: {epoch_time:.2f}s")
        print("=" * 30)

    evaluate(model, test_loader)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    plot_metrics(train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, val_f1_hist)


if __name__ == "__main__":
    main()
