# Imports
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Rice Type Classification using PyTorch
#
# Description:
# This project implements a feedforward neural network in PyTorch for binary
# classification of rice varieties. The workflow includes data loading and
# preprocessing, train/validation/test splitting, feature standardization,
# mini-batch data loading, neural network training, validation, final testing,
# metric visualization, and model checkpointing.
#
# The model consists of an input layer, a ReLU-activated hidden layer, dropout
# regularization, and a sigmoid output layer for binary classification.
# Binary Cross-Entropy (BCE) is used as the loss function and Adam is used
# for gradient-based optimization.
#
# Dataset:
# Rice Type Classification
# Kaggle: https://www.kaggle.com/datasets/mssmartypants/rice-type-classification
# The dataset is provided locally as "riceClassification.csv".


class RiceDataset(Dataset):
    def __init__(self, X, y, device):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y.values, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p=0.3):
        super(MyModel, self).__init__()
        self.input_layer = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    print(df.head())

    # Inspect dataset structure before preprocessing.
    for col in df.columns:
        print(f"Column: {col}, Unique Values: {df[col].nunique()}, Data Type: {df[col].dtype}")

    df.dropna(inplace=True)
    df.drop(columns=['id'], inplace=True)

    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y


def split_and_scale(X, y):
    # Keep validation and test data separate from the training process.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Fit the scaler only on training data to prevent data leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    total_acc = 0
    for inputs, labels in loader:
        optimizer.zero_grad()
        prediction = model(inputs).squeeze(1)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        acc = ((prediction.round() == labels).sum().item()) / len(labels)
        total_acc += acc

    return total_loss / len(loader), total_acc / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for inputs, labels in loader:
            prediction = model(inputs).squeeze(1)
            loss = criterion(prediction, labels)

            total_loss += loss.item()
            acc = ((prediction.round() == labels).sum().item()) / len(labels)
            total_acc += acc

    return total_loss / len(loader), total_acc / len(loader)


def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))

    axs[0].plot(train_losses, label="Train Loss")
    axs[0].plot(val_losses, label="Val Loss")
    axs[0].set_title("Loss over Epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot([a * 100 for a in train_accs], label="Train Accuracy")
    axs[1].plot([a * 100 for a in val_accs], label="Val Accuracy")
    axs[1].set_title("Accuracy over Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_ylim(0, 100)
    axs[1].legend()

    plt.show()


def main():
    # Use GPU acceleration when available; otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y = load_and_preprocess_data("riceClassification.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale(X, y)

    training_data = RiceDataset(X_train, y_train, device)
    val_data = RiceDataset(X_val, y_val, device)
    testing_data = RiceDataset(X_test, y_test, device)

    # Mini-batches allow the model to update parameters incrementally.
    train_loader = DataLoader(training_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    test_loader = DataLoader(testing_data, batch_size=8, shuffle=False)

    HIDDEN_LAYER_SIZE = 32
    input_dim = X_train.shape[1]
    hidden_dim = HIDDEN_LAYER_SIZE
    epochs = 30

    model = MyModel(input_dim, hidden_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    print(summary(model, input_size=(input_dim,)))

    total_loss_train_plot = []
    total_acc_train_plot = []
    total_loss_val_plot = []
    total_acc_val_plot = []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        total_loss_train_plot.append(train_loss)
        total_acc_train_plot.append(train_acc)
        total_loss_val_plot.append(val_loss)
        total_acc_val_plot.append(val_acc)

        print(f'''Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, 
            Train Acc: {train_acc*100:.2f}%, 
            Val Loss: {val_loss:.4f}, 
            Val Acc: {val_acc*100:.2f}%''')

    # Evaluate once on the held-out test set after training is complete.
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f'''Test Loss: {test_loss:.4f}, 
        Test Acc: {test_acc*100:.2f}%''')

    # Save the trained parameters for later inference or analysis.
    torch.save(model.state_dict(), "rice_model.pth")

    plot_metrics(total_loss_train_plot, total_loss_val_plot, total_acc_train_plot, total_acc_val_plot)


if __name__ == "__main__":
    main()
