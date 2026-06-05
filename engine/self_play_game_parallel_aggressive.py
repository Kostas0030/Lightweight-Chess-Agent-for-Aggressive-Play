# CURRENT BEST

# self_play_batch_no_counter.py
import os
import random
import pickle
from collections import namedtuple

import chess
from chess import syzygy, polyglot
import numpy as np
import torch
from tqdm import tqdm

from syzygy import tablebase_best_move
from mcts_game_parallel_aggressive import MCTSBatch, Node
from auxiliary_func import initialize_policy_model, initialize_value_model

# =========================
# Data container
# =========================
GameData = namedtuple("GameData", ["position", "pi", "z"])

# =========================
# Worker initialization (no counter)
# =========================
_worker_runner = None

def _init_worker(runner):
    global _worker_runner
    _worker_runner = runner
    _worker_runner._init_models()

def _play_worker(start_positions):
    return _worker_runner.play_batch(start_positions)

# =========================
# Self-Play Runner (Batch, No Counter)
# =========================
class SelfPlayRunnerBatch:
    def __init__(
        self,
        policy_path: str,
        value_path: str,
        save_path: str,
        num_simulations: int = 100,
        batch_size: int = 16,
    ):
        self.policy_path = policy_path
        self.value_path = value_path
        self.save_path = save_path
        self.num_simulations = num_simulations
        self.batch_size = batch_size

        self.device = None
        self.policy_model = None
        self.value_model = None
        self.int_to_move = None
        self.move_to_index = None
        self.mcts = None

    # -------------------------
    # Initialize models (per worker)
    # -------------------------
    def _init_models(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_model, self.int_to_move = initialize_policy_model(
            self.policy_path, self.device
        )

        self.move_to_index = {v: k for k, v in self.int_to_move.items()}

        self.value_model = initialize_value_model(
            self.value_path, self.device
        )

        self.book = polyglot.open_reader("data/opening_books/gambit_opening_book.bin")
        self.tb = syzygy.open_tablebase("data/tablebase")

        self.mcts = MCTSBatch(
            policy_model=self.policy_model,
            value_model=self.value_model,
            device=self.device,
            int_to_move=self.int_to_move,
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
    # Play batch of games
    # -------------------------
    def play_batch(self, start_positions, lambda_fast_win=0.5, contempt=0.1):
        batch_boards = [
            chess.Board(random.choice(start_positions))
            for _ in range(self.batch_size)
        ]

        root_nodes = [Node(board) for board in batch_boards]
        start_fens = [board.fen() for board in batch_boards]  # track starting turn

        finished = [False] * self.batch_size
        game_histories = [[] for _ in range(self.batch_size)]
        ply_counts = [0] * self.batch_size  # track ply per game

        while not all(finished):
            mcts_indices = []
            mcts_roots = []

            for i in range(self.batch_size):
                if finished[i]:
                    continue

                root = root_nodes[i]

                # ---------- Opening Book ----------
                if self.book is not None and root.board.fullmove_number < 5:
                    while root.board.fullmove_number < 5:
                        book_move = self.sample_polyglot_move(root.board)
                        if book_move is not None:
                            pi = np.zeros(len(self.int_to_move), dtype=np.float32)
                            uci = book_move.uci()
                            pi[self.move_to_index[uci]] = 1.0

                            game_histories[i].append((root.board.fen(), pi))
                            ply_counts[i] += 1

                            root.board.push(book_move)
                            root_nodes[i] = Node(root.board)
                        else:
                            break

                # ---------- Tablebase ----------
                if self.tb is not None and len(root_nodes[i].board.piece_map()) <= 5:
                    while not root.board.is_game_over():
                        root_nodes[i].board.castling_rights = 0
                        best_move = tablebase_best_move(root_nodes[i].board, self.tb)
                        if best_move is not None:
                            pi = np.zeros(len(self.int_to_move), dtype=np.float32)
                            uci = best_move.uci()
                            if uci.endswith(("r", "b", "n")):
                                uci = uci[:-1] + "q"

                            pi[self.move_to_index[uci]] = 1.0

                            game_histories[i].append((root.board.fen(), pi))
                            ply_counts[i] += 1

                            root.board.push(best_move)
                            root_nodes[i] = Node(root.board)

                            if root.board.is_game_over():
                                finished[i] = True
                        else:
                            break
                else:
                    mcts_indices.append(i)
                    mcts_roots.append(root_nodes[i])

            # --------------------------------
            # Run MCTS ONLY on remaining games
            # --------------------------------
            if mcts_roots:
                self.mcts.search_batch(mcts_roots, self.num_simulations)

                for i, root in zip(mcts_indices, mcts_roots):
                    pi = self.mcts.get_pi(root)
                    non_zero_idx = np.nonzero(pi)[0]

                    if len(non_zero_idx) == 0:
                        finished[i] = True
                        game_histories[i].append((root.board.fen(), np.zeros_like(pi)))
                        continue

                    probs = pi[non_zero_idx]
                    probs /= probs.sum()
                    move_idx = np.random.choice(non_zero_idx, p=probs)
                    move = chess.Move.from_uci(self.int_to_move[move_idx])

                    game_histories[i].append((root.board.fen(), pi))
                    ply_counts[i] += 1  # increment ply for this game
                    root.board.push(move)

                    child = root.children.get(move)
                    child.parent = None
                    root_nodes[i] = child

                    if root.board.is_game_over():
                        finished[i] = True

        # --------------------------------
        # Compute rewards using ply length
        # --------------------------------
        final_data = []
        for i, history in enumerate(game_histories):
            result = root_nodes[i].board.result()
            L = max(1, ply_counts[i])  # this game's specific ply count

            r_white = 0.0
            r_black = 0.0
            if result == "1-0":
                r_white = 1.0 + lambda_fast_win / L
                r_black = -1.0
            elif result == "0-1":
                r_white = -1.0
                r_black = 1.0 + lambda_fast_win / L
            else:
                r_white = -contempt
                r_black = -contempt

            # assign rewards alternating per ply
            first_turn = chess.Board(start_fens[i]).turn
            rewards = [r_white, r_black] if first_turn == chess.WHITE else [r_black, r_white]

            for j, (fen, pi) in enumerate(history):
                final_data.append(GameData(fen, pi, rewards[j % 2]))

        return final_data

    # -------------------------
    # Main loop
    # -------------------------
    def run(self, num_samples: int, num_processes: int = 2):
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

        print(f"Using {num_processes} processes")

        from multiprocessing import Pool  # local import

        with Pool(
            processes=num_processes,
            initializer=_init_worker,
            initargs=(self,)
        ) as pool:

            pbar = tqdm(total=num_samples, desc="Generating positions")

            while len(new_data) < num_samples:
                results = pool.map(_play_worker, [[chess.STARTING_FEN]] * num_processes)

                for batch_data in results:
                    new_data.extend(batch_data)
                    pbar.update(len(batch_data))

                    # Periodic save
                    if len(new_data) % save_interval < len(batch_data):
                        with open(self.save_path, "wb") as f:
                            pickle.dump(existing_data + new_data, f)
                        print(f"\nSaved {len(existing_data) + len(new_data)} samples")

                    if len(new_data) >= num_samples:
                        break

            pbar.close()

        with open(self.save_path, "wb") as f:
            pickle.dump(existing_data + new_data, f)

        print(f"Final save: {len(existing_data) + len(new_data)} samples")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    policy_path = "model/POLICY_MODEL_10EPOCHS_V3.pth"
    value_path = "model/VALUE_MODEL_10EPOCHS_V3.pth"
    save_path = "data/self_play_data/aggressive/lambda_0.5/V3_100_1m.pkl"

    runner = SelfPlayRunnerBatch(
        policy_path,
        value_path,
        save_path,
        num_simulations=100,
        batch_size=128,
    )

    runner.run(
        num_samples=200_000,
        num_processes=16
    )