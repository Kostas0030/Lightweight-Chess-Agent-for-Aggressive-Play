import sys
import os
import chess
import torch
from mcts_tt_tb import MCTS
from policy_network import PolicyNet
from value_network import ValueNet
import pickle

# ============================================================
# Robust paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

policy_path = os.path.join(MODEL_DIR, "POLICY_MODEL_10EPOCHS_V0.pth")
value_path = os.path.join(MODEL_DIR, "VALUE_MODEL_10EPOCHS_V0.pth")
move_to_int_path = os.path.join(MODEL_DIR, "move_to_int")

# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Model loaders
# ============================================================
def initialize_policy_model(policy_model_path, device, move_to_int_path):
    with open(move_to_int_path, "rb") as file:
        move_to_int = pickle.load(file)

    policy_model = PolicyNet(num_classes=len(move_to_int))
    policy_model.load_state_dict(torch.load(policy_model_path, map_location=device))
    policy_model.to(device)
    policy_model.eval()

    int_to_move = {v: k for k, v in move_to_int.items()}
    return policy_model, int_to_move


def initialize_value_model(value_model_path, device):
    value_model = ValueNet()
    value_model.load_state_dict(torch.load(value_model_path, map_location=device))
    value_model.to(device)
    value_model.eval()
    return value_model

# ============================================================
# UCI loop
# ============================================================
def main():
    board = chess.Board()
    mcts = None

    try:
        while True:
            command = sys.stdin.readline().strip()
            if not command:
                continue

            if command == "uci":
                # Respond immediately — do NOT load models here
                print("id name MyMCTSEngine")
                print("id author You")
                print("uciok", flush=True)

            elif command == "isready":
                # Load models here — cutechess waits longer for readyok
                print("info string loading models...", flush=True)
                policy_model, int_to_move = initialize_policy_model(policy_path, device, move_to_int_path)
                value_model = initialize_value_model(value_path, device)
                mcts = MCTS(
                    policy_model=policy_model,
                    value_model=value_model,
                    device=device,
                    int_to_move=int_to_move,
                    tablebase=None
                )
                print("readyok", flush=True)

            elif command == "ucinewgame":
                board = chess.Board()

            elif command.startswith("position"):
                parts = command.split()
                if parts[1] == "startpos":
                    board = chess.Board()
                    if "moves" in parts:
                        moves = parts[parts.index("moves") + 1:]
                    else:
                        moves = []
                elif parts[1] == "fen":
                    fen = " ".join(parts[2:8])
                    board = chess.Board(fen)
                    if "moves" in parts:
                        moves = parts[parts.index("moves") + 1:]
                    else:
                        moves = []
                else:
                    continue

                for move in moves:
                    try:
                        board.push_uci(move)
                    except:
                        print(f"info string illegal move {move}", flush=True)

            elif command.startswith("go"):
                if mcts is None:
                    print("info string engine not ready", flush=True)
                    print("bestmove 0000", flush=True)
                    continue

                print("info string thinking...", flush=True)
                try:
                    best_move, _, _ = mcts.search(board, num_simulations=200)
                    if best_move is None:
                        best_move = "0000"
                except Exception as e:
                    print(f"info string search error: {e}", flush=True)
                    best_move = "0000"

                print(f"bestmove {best_move}", flush=True)

            elif command == "quit":
                break

    except Exception as e:
        print(f"info string fatal error: {e}", flush=True)

if __name__ == "__main__":
    main()