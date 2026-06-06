import io
import chess
import chess.pgn
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from auxiliary_func import initialize_policy_model, initialize_value_model
from syzygy import tablebase_best_move
from mcts_tt_tb import MCTS
from chess import syzygy

# ============================================================
# Globals inside each worker (REQUIRED for multiprocessing)
# ============================================================
_policy_new = None
_policy_old = None
_value_new = None
_value_old = None
_int_to_move = None
_device = None
_tb = None
_use_tb_flag = True
_move_cutoff = 200


# ============================================================
# Worker initializer (TOP-LEVEL — DO NOT MOVE)
# ============================================================
def worker_init(
    policy_new_path,
    policy_old_path,
    value_new_path,
    value_old_path,
    tb_path="data/tablebase",
    use_cuda=False,
    use_tb=True,
    move_cutoff=200,
):
    global _policy_new, _policy_old, _value_new, _value_old
    global _int_to_move, _device, _tb, _use_tb_flag, _move_cutoff

    _use_tb_flag = use_tb
    _move_cutoff = move_cutoff

    _device = torch.device(
        "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    )

    _policy_new, _int_to_move = initialize_policy_model(policy_new_path, _device)
    _policy_old, _ = initialize_policy_model(policy_old_path, _device)
    _value_new = initialize_value_model(value_new_path, _device)
    _value_old = initialize_value_model(value_old_path, _device)

    if use_tb:
        _tb = syzygy.open_tablebase(tb_path)

    print(
        f"Worker initialized | Device: {_device} | "
        f"Tablebase: {_use_tb_flag} | Cutoff: {_move_cutoff}"
    )


# ============================================================
# Move selection
# ============================================================
def select_move(
    board,
    policy_model,
    value_model,
    int_to_move,
    device,
    tb,
    use_tb_flag,
    num_simulations,
):
    if use_tb_flag and len(board.piece_map()) <= 5:
        move = tablebase_best_move(board, tb)
        if move is not None:
            return move

    mcts = MCTS(
        policy_model=policy_model,
        value_model=value_model,
        device=device,
        int_to_move=int_to_move,
        tablebase=tb,
    )

    _, _, pi = mcts.search(board, num_simulations)

    legal_moves = list(board.legal_moves)
    uci_to_idx = {int_to_move[i]: i for i in range(len(pi))}

    legal_indices, legal_probs = [], []
    for move in legal_moves:
        uci = move.uci()
        if uci in uci_to_idx:
            idx = uci_to_idx[uci]
            legal_indices.append(idx)
            legal_probs.append(pi[idx])

    if not legal_probs:
        return legal_moves[0]

    legal_probs = np.array(legal_probs, dtype=np.float64)
    legal_probs /= legal_probs.sum()

    chosen = np.random.choice(len(legal_indices), p=legal_probs)
    return chess.Move.from_uci(int_to_move[legal_indices[chosen]])


# ============================================================
# Material evaluation for cutoff
# ============================================================
def evaluate_material(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    score = 0
    for pt, val in piece_values.items():
        score += len(board.pieces(pt, chess.WHITE)) * val
        score -= len(board.pieces(pt, chess.BLACK)) * val

    if score > 0:
        return "1-0"
    elif score < 0:
        return "0-1"
    return "1/2-1/2"


# ============================================================
# Single game task (TOP-LEVEL)
# ============================================================
def play_game_task(args):
    i, num_simulations = args

    global _policy_new, _policy_old, _value_new, _value_old
    global _int_to_move, _device, _tb, _use_tb_flag, _move_cutoff

    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    if i % 2 == 0:
        white_policy, white_value = _policy_new, _value_new
        black_policy, black_value = _policy_old, _value_old
        game.headers["White"] = "Model_New"
        game.headers["Black"] = "Model_Old"
    else:
        white_policy, white_value = _policy_old, _value_old
        black_policy, black_value = _policy_new, _value_new
        game.headers["White"] = "Model_Old"
        game.headers["Black"] = "Model_New"

    move_counter = 0

    while not board.is_game_over():
        if move_counter >= _move_cutoff:
            result = evaluate_material(board)
            game.headers["Result"] = result
            pgn_str = str(game)
            return (
                1 if result == "1-0" else -1 if result == "0-1" else 0,
                pgn_str,
            )

        if board.turn == chess.WHITE:
            move = select_move(
                board,
                white_policy,
                white_value,
                _int_to_move,
                _device,
                _tb,
                _use_tb_flag,
                num_simulations,
            )
        else:
            move = select_move(
                board,
                black_policy,
                black_value,
                _int_to_move,
                _device,
                _tb,
                _use_tb_flag,
                num_simulations,
            )

        board.push(move)
        node = node.add_variation(move)
        move_counter += 1

    result = board.result()
    game.headers["Result"] = result

    if i % 2 == 0:
        outcome = 1 if result == "1-0" else -1 if result == "0-1" else 0
    else:
        outcome = 1 if result == "0-1" else -1 if result == "1-0" else 0

    return outcome, str(game)


# ============================================================
# CLASS: ModelEvaluator
# ============================================================
class ModelEvaluator:
    def __init__(
        self,
        policy_new_path,
        policy_old_path,
        value_new_path,
        value_old_path,
        n_games=100,
        num_simulations=200,
        tb_path="data/tablebase",
        use_cuda=False,
        use_tablebase=True,
        move_cutoff=200,
        threshold=0.55,
        save_pgn=False,
        pgn_path=None,
    ):
        self.policy_new_path = policy_new_path
        self.policy_old_path = policy_old_path
        self.value_new_path = value_new_path
        self.value_old_path = value_old_path
        self.n_games = n_games
        self.num_simulations = num_simulations
        self.tb_path = tb_path
        self.use_cuda = use_cuda
        self.use_tablebase = use_tablebase
        self.move_cutoff = move_cutoff
        self.threshold = threshold
        self.save_pgn = save_pgn
        self.pgn_path = pgn_path

        self.scores = None
        self.games = None

    def run(self):
        num_processes = cpu_count()
        print(
            f"Using {num_processes} processes | "
            f"Tablebase: {self.use_tablebase} | "
            f"Cutoff: {self.move_cutoff}"
        )

        init_args = (
            self.policy_new_path,
            self.policy_old_path,
            self.value_new_path,
            self.value_old_path,
            self.tb_path,
            self.use_cuda,
            self.use_tablebase,
            self.move_cutoff,
        )

        tasks = [(i, self.num_simulations) for i in range(self.n_games)]

        results = []
        games = []

        plies_per_win = defaultdict(list)

        with Pool(
            processes=num_processes,
            initializer=worker_init,
            initargs=init_args,
        ) as pool:
            for idx, (outcome, pgn_str) in enumerate(
                tqdm(
                    pool.imap(play_game_task, tasks),
                    total=self.n_games,
                    desc="Evaluating",
                ),
                1,
            ):
                game = chess.pgn.read_game(io.StringIO(pgn_str))
                results.append((outcome, game))
                games.append(game)

                plies = len(list(game.mainline_moves()))

                if outcome == 1:
                    plies_per_win["new"].append(plies)
                elif outcome == -1:
                    plies_per_win["old"].append(plies)

                if idx % 10 == 0:
                    self._print_intermediate(results, idx, plies_per_win)

        self.scores = {1: 0, 0: 0, -1: 0}
        for outcome, _ in results:
            self.scores[outcome] += 1

        self.games = games

        print("\n=== Aggression Metrics ===")
        print(
            f"Avg plies per NEW win: "
            f"{np.mean(plies_per_win['new']) if plies_per_win['new'] else 0:.2f}"
        )
        print(
            f"Avg plies per OLD win: "
            f"{np.mean(plies_per_win['old']) if plies_per_win['old'] else 0:.2f}"
        )

        if self.save_pgn:
            self._save_pgn()

        return self.is_better(), self.scores, self.games

    def _save_pgn(self):
        if not self.pgn_path:
            raise ValueError("pgn_path must be provided when save_pgn=True")

        print(f"\nSaving PGN to: {self.pgn_path}")
        with open(self.pgn_path, "w", encoding="utf-8") as f:
            for game in self.games:
                print(game, file=f)
                print(file=f)

    def _print_intermediate(self, results, idx, plies_per_win):
        scores = {1: 0, 0: 0, -1: 0}
        for outcome, _ in results:
            scores[outcome] += 1

        avg_new = np.mean(plies_per_win["new"]) if plies_per_win["new"] else 0.0
        avg_old = np.mean(plies_per_win["old"]) if plies_per_win["old"] else 0.0

        print(f"\n--- Results after {idx} games ---")
        print(f"New wins: {scores[1]}, Draws: {scores[0]}, Old wins: {scores[-1]}")
        print(f"New plies: {avg_new:.2f}, Old plies: {avg_old:.2f}")

    def is_better(self):
        wins = self.scores[1]
        losses = self.scores[-1]
        total = wins + losses

        if total == 0:
            return False

        win_rate = wins / total
        print(f"\nWin rate (new model): {win_rate:.3f}")
        return win_rate >= self.threshold


# ============================================================
# MAIN
# ============================================================
def main():
    evaluator = ModelEvaluator(
        policy_new_path="model/POLICY_MODEL_10EPOCHS_V4.pth",
        policy_old_path="model/POLICY_MODEL_10EPOCHS_V3.pth",
        value_new_path="model/VALUE_MODEL_10EPOCHS_V4.pth",
        value_old_path="model/VALUE_MODEL_10EPOCHS_V3.pth",
        n_games=500,
        num_simulations=100,
        move_cutoff=400,
        threshold=0.55,
        use_cuda=False,
        use_tablebase=False,
        save_pgn=True,
        pgn_path="games.pgn",
    )

    is_better, _, _ = evaluator.run()
    print("\nAccept new model:", is_better)


if __name__ == "__main__":
    main()
