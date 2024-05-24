import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Set seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# Check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15

# Image size and number of classes
IMG_SIZE = 32
N_CLASSES = 10

# Function to get accuracy
def get_accuracy(model, data_loader, device):
    correct_pred = 0
    n = 0
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)
            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)
            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()
    return correct_pred.float() / n

# Function to plot losses
def plot_losses(train_losses, valid_losses):
    plt.style.use('seaborn')
    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_losses, color='blue', label='Training loss')
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", xlabel='Epoch', ylabel='Loss')
    ax.legend()
    fig.show()
    plt.style.use('default')

# Function for the training step of the training loop
def train(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for X, y_true in train_loader:
        optimizer.zero_grad()
        X = X.to(device)
        y_true = y_true.to(device)
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)
        loss.backward()
        optimizer.step()
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

# Function for the validation step of the training loop
def validate(valid_loader, model, criterion, device):
    model.eval()
    running_loss = 0
    for X, y_true in valid_loader:
        X = X.to(device)
        y_true = y_true.to(device)
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)
    epoch_loss = running_loss / len(valid_loader.dataset)
    return model, epoch_loss

# Function defining the entire training loop
def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    for epoch in range(0, epochs):
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    plot_losses(train_losses, valid_losses)

    return model, optimizer, (train_losses, valid_losses)

# Define transforms
transforms = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                 transforms.ToTensor()])

# Download and create datasets
train_dataset = datasets.MNIST(root='mnist_data',
                               train=True,
                               transform=transforms,
                               download=True)

valid_dataset = datasets.MNIST(root='mnist_data',
                               train=False,
                               transform=transforms)

# Define the data loaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False)

# Function to visualize predictions
def visualize_predictions(model, test_loader, device, num_images=10):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(inputs.size(0)):
                predictions.append((inputs[i], predicted[i], labels[i]))

    # Visualize a subset of predictions
    ROW_IMG = 5
    N_ROWS = 2
    fig = plt.figure()

    for index in range(1, num_images + 1):
        plt.subplot(N_ROWS, ROW_IMG, index)
        plt.axis('off')

        # Extract data for visualization
        inputs, predicted, true_label = predictions[index - 1]
        img = inputs[0].cpu().numpy().squeeze()

        # Display the image
        plt.imshow(img, cmap='gray_r')

        # Show predicted and true labels
        title = f'Pred: {predicted.item()}\nTrue: {true_label.item()}'
        plt.title(title, fontsize=7)

    fig.suptitle('Model Predictions on Test Data')
    plt.show()

# LeNet-5-like architecture
class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(16, 120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, n_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

# Initialize the model, optimizer, and criterion
model = LeNet5(N_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Train the model
model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)

# Visualize predictions on test data
visualize_predictions(model, valid_loader, DEVICE)

