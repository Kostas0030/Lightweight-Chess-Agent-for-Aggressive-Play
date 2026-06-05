import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt  # ADDED

from auxiliary_func import load_games_from_directory, prepare_policy_training_data
from dataset import ChessDataset
from policy_network import PolicyNet


"""
Function to initialize all the training components -> data, dataLoader, device, model, criterion, optimizer

input: X (board states encoded), y (target moves encoded), num_classes
returns: training components
"""
def initialize_policy_training_components(X, y, num_classes, batch_size=64):
    dataset = ChessDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    model = PolicyNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    return model, dataloader, criterion, optimizer, device


"""
Function that trains the model

input: training components
returns: trained model, list of epoch losses
"""
def train_policy_net(model, dataloader, criterion, optimizer, device, start_epoch=0, num_epochs=10):
    epoch_losses = []  # ADDED

    for epoch in range(start_epoch, start_epoch + num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)  # ADDED

        duration = time.time() - start_time
        print(
            f'Epoch {epoch + 1}/{start_epoch + num_epochs}, '
            f'Loss: {avg_loss:.4f}, '
            f'Time: {int(duration // 60)}m{int(duration % 60)}s'
        )

    return model, epoch_losses  # MODIFIED RETURN


"""
Function that saves the trained model and the move mapping ("move" -> integer)

input: model, move_to_int -> Dictionary{"move" -> integer}, model_path, mapping_path
returns: -
"""
def save_model_and_mapping(model, move_to_int, model_path, mapping_path):
    torch.save(model.state_dict(), model_path)
    with open(mapping_path, "wb") as file:
        pickle.dump(move_to_int, file)
    print(f"Model saved to: {model_path}")
    print(f"Mapping saved to: {mapping_path}")


def main():
    print("Loading data...")
    games = load_games_from_directory("data/database", limit=28)
    print("Loading finished!")

    print("Preparing tensors...")
    X_tensor, y_tensor, num_classes, move_to_int = prepare_policy_training_data(
        games, max_samples=5_000_000
    )
    print("Preparing tensors finished!")

    model, dataloader, criterion, optimizer, device = initialize_policy_training_components(
        X_tensor, y_tensor, num_classes
    )

    print("Training policy network...")
    policy_model, losses = train_policy_net(
        model, dataloader, criterion, optimizer, device,
        start_epoch=0, num_epochs=10
    )
    print("Training finished!")

    # -------- PLOT & SAVE LOSS --------
    epochs = list(range(1, len(losses) + 1))

    plt.figure()
    plt.plot(epochs, losses, label="Policy Network Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Policy Network Training Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("policy_network_loss.png")
    plt.show()
    # ---------------------------------

    save_model_and_mapping(
        policy_model,
        move_to_int,
        "models/POLICY_MODEL_10EPOCHS_V0_EXP.pth",
        "models/move_to_int"
    )


if __name__ == "__main__":
    main()
