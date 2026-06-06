# self_play.py
import os
import random
import pickle
import time
from collections import namedtuple

import chess
from chess import Board, syzygy
import numpy as np
import torch
from tqdm import tqdm

from initial_mcts_leaf_par import MCTS
from auxiliary_func import initialize_policy_model, initialize_value_model


# =========================
# Data container
# =========================
GameData = namedtuple("GameData", ["position", "pi", "z"])


# =========================
# SELF-PLAY CLASS
# =========================
class SelfPlayRunner:
    def __init__(
        self,
        policy_path: str,
        value_path: str,
        save_path: str,
        num_simulations: int = 100,
    ):
        self.policy_path = policy_path
        self.value_path = value_path
        self.save_path = save_path
        self.num_simulations = num_simulations

        self.device = None
        self.policy_model = None
        self.value_model = None
        self.int_to_move = None
        self.mcts = None
        #self.tb = None

        self._init_models()

    # -------------------------
    # Model + MCTS initialization
    # -------------------------
    def _init_models(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_model, self.int_to_move = initialize_policy_model(
            self.policy_path, self.device
        )
        self.value_model = initialize_value_model(
            self.value_path, self.device
        )

        #self.tb = syzygy.open_tablebase("data/tablebase")

        self.mcts = MCTS(
            policy_model=self.policy_model,
            value_model=self.value_model,
            device=self.device,
            int_to_move=self.int_to_move,
            #tablebase=self.tb,
            c_puct=1.5,
        )

    # -------------------------
    # Play one game
    # -------------------------
    def play_from_position(self, start_fen: str):
        board = Board(start_fen)
        game_data = []

        while not board.is_game_over():
            position = board.fen()
            _, _, pi = self.mcts.search(board, self.num_simulations)

            game_data.append((position, pi))

            non_zero_idx = np.nonzero(pi)[0]
            probs = pi[non_zero_idx]
            probs /= probs.sum()

            move_idx = np.random.choice(non_zero_idx, p=probs)
            move = chess.Move.from_uci(self.int_to_move[move_idx])

            board.push(move)

        # --- Compute reward ---
        result = board.result()
        if result == "1-0":
            r = 1
        elif result == "0-1":
            r = -1
        else:
            r = 0

        final_data = []
        player = 1 if Board(start_fen).turn == chess.WHITE else -1

        for position, pi in game_data:
            z = r * player
            final_data.append(GameData(position, pi, z))
            player *= -1

        return final_data

    # -------------------------
    # Time-based self-play loop
    # -------------------------
    def run(self, max_runtime_seconds: int):
        start_time = time.time()
        end_time = start_time + max_runtime_seconds

        existing_data = []
        if os.path.exists(self.save_path):
            with open(self.save_path, "rb") as f:
                try:
                    existing_data = pickle.load(f)
                    print(f"Loaded {len(existing_data)} existing samples")
                except Exception:
                    existing_data = []

        new_data = []
        save_interval = 100_000

        with open("data/filtered_positions_from_database.pkl", "rb") as f:
            start_positions = pickle.load(f)

        pbar = tqdm(desc="Self-play (1 hour)", unit="positions")

        while time.time() < end_time:
            game = self.play_from_position(random.choice(start_positions))
            new_data.extend(game)
            pbar.update(len(game))

            if len(new_data) % save_interval < len(game):
                with open(self.save_path, "wb") as f:
                    pickle.dump(existing_data + new_data, f)
                print(f"\nSaved {len(existing_data) + len(new_data)} samples")

        pbar.close()

        with open(self.save_path, "wb") as f:
            pickle.dump(existing_data + new_data, f)

        elapsed = time.time() - start_time

        # =========================
        # FINAL STATISTICS
        # =========================
        print("\n===================================")
        print("SELF-PLAY SUMMARY")
        print("===================================")
        print(f"Elapsed time       : {elapsed / 3600:.2f} hours")
        print(f"Positions generated: {len(new_data)}")

        """print()
        self.mcts.print_tt_stats()"""


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    policy_path = "model/POLICY_MODEL_10EPOCHS_V0.pth"
    value_path = "model/VALUE_MODEL_10EPOCHS_V0.pth"
    save_path = "data/self_play_data/aggressive/lambda_0.5/self_play_V0_optimization.pkl"

    runner = SelfPlayRunner(
        policy_path=policy_path,
        value_path=value_path,
        save_path=save_path,
        num_simulations=200,
    )

    # Run exactly 1 hour
    runner.run(max_runtime_seconds=600)
