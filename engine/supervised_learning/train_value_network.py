import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt  # ADDED

from dataset import ChessDataset
from value_network import ValueNet
from auxiliary_func import load_games_from_directory, prepare_value_training_data


def initialize_value_training_components(X_tensor, y_tensor, batch_size=256, lr=0.001):
    dataset = ChessDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ValueNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, dataloader, criterion, optimizer, device


def train_value_net(model, dataloader, criterion, optimizer, device, num_epochs, start_epoch=0):
    epoch_losses = []  # ADDED

    for epoch in range(start_epoch, start_epoch + num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        duration = time.time() - start_time
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)  # ADDED

        current_epoch = epoch + 1
        print(
            f'Epoch {current_epoch}/{start_epoch + num_epochs}, '
            f'Loss: {avg_loss:.4f}, '
            f'Time: {int(duration // 60)}m{int(duration % 60)}s'
        )

    return epoch_losses  # ADDED


"""
Function that saves the trained model

input: model, model_path
returns: -
"""
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


def main():
    print("Loading data...")
    games = load_games_from_directory("data/database", limit=28)
    print("Loading finished!")

    print("Preparing tensors...")
    X_tensor, y_tensor = prepare_value_training_data(games, max_samples=5_000_000)
    print("Preparing tensors finished!")
        
    model, dataloader, criterion, optimizer, device = initialize_value_training_components(
        X_tensor, y_tensor
    )

    print("Training value network...")
    losses = train_value_net(
        model, dataloader, criterion, optimizer, device,
        num_epochs=10, start_epoch=0
    )
    print("Training finished!")

    # -------- PLOT & SAVE LOSS --------
    epochs = list(range(1, len(losses) + 1))

    plt.figure()
    plt.plot(epochs, losses, label="Value Network Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Value Network Training Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("value_network_loss.png")
    plt.show()
    # ---------------------------------

    save_model(model, "models/VALUE_MODEL_10EPOCHS_V0.pth")


if __name__ == "__main__":
    main()
