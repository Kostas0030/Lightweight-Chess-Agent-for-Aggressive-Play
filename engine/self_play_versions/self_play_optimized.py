import os
import random
import pickle
import subprocess
from multiprocessing import Pool, cpu_count
from collections import namedtuple

import chess
from chess import Board, syzygy, polyglot
import numpy as np
import torch
from tqdm import tqdm

from mcts_optimized import MCTS
from auxiliary_func import initialize_policy_model, initialize_value_model
from syzygy import tablebase_best_move


# =========================
# Data container
# =========================
GameData = namedtuple("GameData", ["position", "pi", "z"])


# =========================
# GLOBALS FOR WORKERS
# (required on Windows)
# =========================
_worker_runner = None


def _init_worker(runner):
    
    #Called once per worker process.
    #Initializes models, MCTS, and tablebases.
    
    global _worker_runner
    _worker_runner = runner
    _worker_runner.init_worker()


def _play_worker(start_fen):
    
    #Worker entry point.
    
    return _worker_runner.play_from_position(start_fen)


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

        # Loaded per worker
        self.device = None
        self.policy_model = None
        self.value_model = None
        self.int_to_move = None
        self.book = None
        self.tb = None
        self.mcts = None

    # -------------------------
    # Worker initialization
    # -------------------------
    def init_worker(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_model, self.int_to_move = initialize_policy_model(
            self.policy_path, self.device
        )
        self.value_model = initialize_value_model(
            self.value_path, self.device
        )

        self.book = polyglot.open_reader("data/opening_books/gambit_opening_book.bin")

        self.tb = syzygy.open_tablebase("data/tablebase")

        self.mcts = MCTS(
            policy_model=self.policy_model,
            value_model=self.value_model,
            device=self.device,
            int_to_move=self.int_to_move,
            tablebase=self.tb,
            c_puct=1.5,
        )

    def sample_polyglot_move(self, board):
        entries = list(self.book.find_all(board))
        if not entries:
            return None

        total_weight = sum(e.weight for e in entries)
        r = np.random.uniform(0, total_weight)

        acc = 0
        for e in entries:
            acc += e.weight
            if acc >= r:
                return e.move

        return None

    # -------------------------
    # Play one game
    # -------------------------
    def play_from_position(self, start_fen: str, lambda_fast_win=0.5, contempt=0.1):
        board = Board(start_fen)
        game_data = []
        ply_count = 0
        move_to_index = {v: k for k, v in self.int_to_move.items()}

        while not board.is_game_over():
            ply_count += 1
            position = board.fen()

            # -------------------------
            # Opening book (ONE-HOT π)
            # -------------------------
            if self.book is not None:
                book_move = self.sample_polyglot_move(board)
                if book_move is not None:
                    pi = np.zeros(len(self.int_to_move), dtype=np.float32)
                    uci = book_move.uci()

                    pi[move_to_index[uci]] = 1.0
                    game_data.append((position, pi))
                    board.push(book_move)
                    continue
            
            # -------------------------
            # Tablebase (ONE-HOT π)
            # -------------------------
            if len(board.piece_map()) <= 5:
                best_move = tablebase_best_move(board, self.tb)
                if best_move is not None:
                    pi = np.zeros(len(self.int_to_move), dtype=np.float32)
                    uci = best_move.uci()
                    if uci.endswith(("r", "b", "n")):
                        uci = uci[:-1] + "q"
                    pi[move_to_index[uci]] = 1.0
                    game_data.append((position, pi))
                    board.push(best_move)
                    continue

            _, _, pi = self.mcts.search(board, self.num_simulations)
            game_data.append((position, pi))

            non_zero_idx = np.nonzero(pi)[0]
            probs = pi[non_zero_idx]
            probs /= probs.sum()
            move_idx = np.random.choice(non_zero_idx, p=probs)
            move = chess.Move.from_uci(self.int_to_move[move_idx])
            board.push(move)

        # --- Compute reward once per side ---
        L = max(1, ply_count)
        result = board.result()

        # White and black rewards vector
        r_white = 0.0
        r_black = 0.0
        if result == "1-0":        # White wins
            r_white = 1.0 + lambda_fast_win / L
            r_black = -1.0
        elif result == "0-1":      # Black wins
            r_white = -1.0
            r_black = 1.0 + lambda_fast_win / L
        else: # Draw (contempt)
            r_white = -contempt
            r_black = -contempt

        # --- Alternate reward assignment without if per move ---
        rewards = [r_white, r_black] if Board(start_fen).turn == chess.WHITE else [r_black, r_white]
        final_data = []

        for i, (position, pi) in enumerate(game_data):
            final_data.append(GameData(position, pi, rewards[i % 2]))

        return final_data


    # -------------------------
    # Main self-play loop
    # -------------------------
    def run(self, num_samples: int):
        existing_data = []
        if os.path.exists(self.save_path):
            with open(self.save_path, "rb") as f:
                try:
                    existing_data = pickle.load(f)
                    print(f"Loaded {len(existing_data)} existing samples")
                except Exception:
                    existing_data = []

        new_data = []
        save_interval = 50_000

        num_processes = cpu_count()
        print(f"Using {num_processes} processes")

        with Pool(
            processes=num_processes,
            initializer=_init_worker,
            initargs=(self,),
        ) as pool:

            pbar = tqdm(total=num_samples, desc="Generating positions")

            while len(new_data) < num_samples:
                batch_positions = [chess.STARTING_FEN] * (cpu_count() * 2)

                results = pool.map(_play_worker, batch_positions)

                for game in results:
                    new_data.extend(game)
                    pbar.update(len(game))

                    if len(new_data) % save_interval < len(game):
                        with open(self.save_path, "wb") as f:
                            pickle.dump(existing_data + new_data, f)
                        print(
                            f"\nSaved {len(existing_data) + len(new_data)} samples"
                        )

                    if len(new_data) >= num_samples:
                        break

            pbar.close()

        with open(self.save_path, "wb") as f:
            pickle.dump(existing_data + new_data, f)

        print(
            f"Final save: {len(existing_data) + len(new_data)} samples"
        )


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    policy_path = "model/POLICY_MODEL_10EPOCHS_V0.pth"
    value_path = "model/VALUE_MODEL_10EPOCHS_V0.pth"
    save_path = "data/self_play_data/aggressive/lambda_0.5/self_play_V0_aggressive_mcts.pkl"

    runner = SelfPlayRunner(
        policy_path=policy_path,
        value_path=value_path,
        save_path=save_path,
        num_simulations=100,
    )

    runner.run(num_samples=1_000_000)

    subprocess.run(
        ["python", "engines/torch/self_play_train.py"],
        check=False,
    )
