from chess import Board
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from auxiliary_func import (
    prepare_input_policy_net,
    prepare_input_value_net,
    initialize_policy_model,
    initialize_value_model
)
from collections import namedtuple
import subprocess


GameData = namedtuple("GameData", ["position", "pi", "z"])


# =========================
# Dataset
# =========================
class SelfPlayDataset(Dataset):
    def __init__(self, pickle_file, policy_length):
        with open(pickle_file, "rb") as f:
            self.data = pickle.load(f)
        self.policy_length = policy_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        position, pi, z = self.data[idx]

        board = Board(position)
        x_policy = prepare_input_policy_net(board).squeeze(0)
        x_value = prepare_input_value_net(board).squeeze(0)

        pi = torch.tensor(pi, dtype=torch.float32)
        z = torch.tensor(z, dtype=torch.float32).unsqueeze(0)

        return x_policy, x_value, pi, z


# =========================
# TRAINER CLASS
# =========================
class SelfPlayTrainer:
    def __init__(
        self,
        policy_model_path,
        value_model_path,
        dataset_path,
        device,
        batch_size=256,
        lr=1e-4,
    ):
        self.device = device
        self.batch_size = batch_size
        self.lr = lr

        self.policy_model, self.int_to_move = initialize_policy_model(
            policy_model_path, device
        )

        self.value_model = initialize_value_model(
            value_model_path, device
        )

        self.dataset = SelfPlayDataset(
            dataset_path, len(self.int_to_move)
        )

        self.policy_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
        )

    # -------------------------
    # Policy training
    # -------------------------
    def train_policy(self, epochs, save_epochs=None, version="V0"):

        criterion = nn.KLDivLoss(reduction="batchmean")
        optimizer = optim.Adam(self.policy_model.parameters(), lr=self.lr)

        self.policy_model.train()
        self.policy_model.to(self.device)

        policy_losses = []

        for epoch in range(epochs):

            total_loss = 0.0

            for x_policy, _, target_pi, _ in tqdm(
                self.policy_loader,
                desc=f"Policy Epoch {epoch+1}/{epochs}",
            ):

                x_policy = x_policy.to(self.device)
                target_pi = target_pi.to(self.device)

                optimizer.zero_grad()

                logits = self.policy_model(x_policy)
                log_probs = torch.log_softmax(logits, dim=1)

                loss = criterion(log_probs, target_pi)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.policy_loader)
            policy_losses.append(avg_loss)

            print(
                f"Policy Epoch {epoch+1} - Loss: {avg_loss:.4f}"
            )

            if save_epochs and (epoch + 1) in save_epochs:
                path = f"model/POLICY_MODEL_{epoch+1}EPOCHS_{version}.pth"
                torch.save(self.policy_model.state_dict(), path)

        # SAVE POLICY LOSSES
        with open(f"data/loss_plots/policy_loss_{version}_100_sims_1m.pkl", "wb") as f:
            pickle.dump(policy_losses, f)

    # -------------------------
    # Value training
    # -------------------------
    def train_value(self, epochs, save_epochs=None, version="V0"):

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.value_model.parameters(), lr=self.lr)

        self.value_model.train()
        self.value_model.to(self.device)

        value_losses = []

        for epoch in range(epochs):

            total_loss = 0.0

            for _, x_value, _, target_z in tqdm(
                self.policy_loader,
                desc=f"Value Epoch {epoch+1}/{epochs}",
            ):

                x_value = x_value.to(self.device)
                target_z = target_z.to(self.device)

                optimizer.zero_grad()

                pred = self.value_model(x_value)

                loss = criterion(pred, target_z)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.value_model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.policy_loader)
            value_losses.append(avg_loss)

            print(
                f"Value Epoch {epoch+1} - Loss: {avg_loss:.4f}"
            )

            if save_epochs and (epoch + 1) in save_epochs:
                path = f"model/VALUE_MODEL_{epoch+1}EPOCHS_{version}.pth"
                torch.save(self.value_model.state_dict(), path)

        # SAVE VALUE LOSSES
        with open(f"data/loss_plots/value_loss_{version}_100_sims_1m.pkl", "wb") as f:
            pickle.dump(value_losses, f)


# =========================
# ENTRY POINT
# =========================
def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    trainer = SelfPlayTrainer(
        policy_model_path="model/POLICY_MODEL_10EPOCHS_V3.pth",
        value_model_path="model/VALUE_MODEL_10EPOCHS_V3.pth",
        dataset_path="data/self_play_data/aggressive/lambda_0.5/V3_100_1m.pkl",
        device=device,
        batch_size=256,
        lr=1e-4,
    )

    trainer.train_policy(
        epochs=20,
        save_epochs=[3, 10, 20],
        version="V4",
    )

    trainer.train_value(
        epochs=20,
        save_epochs=[3, 10, 20],
        version="V4",
    )


if __name__ == "__main__":
    main()

    subprocess.run(
        ["python", "engines/torch/test_network_vs_network.py"],
        check=False,
    )
