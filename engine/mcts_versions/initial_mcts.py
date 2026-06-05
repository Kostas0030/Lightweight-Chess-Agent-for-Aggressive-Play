# mcts.py
import chess
import math
import numpy as np
import torch
from chess import syzygy
from auxiliary_func import (
    prepare_input_policy_net,
    prepare_input_value_net
)

# ============================================================
# Node
# ============================================================

class Node:
    __slots__ = ("board", "parent", "move", "children", "N", "W", "P", "Q")

    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.P = 0.0
        self.Q = 0.0

    def is_expanded(self):
        return bool(self.children)


# ============================================================
# MCTS
# ============================================================

class MCTS:
    def __init__(
        self,
        policy_model,
        value_model,
        device,
        int_to_move,
        c_puct=1.5
    ):
        self.policy_model = policy_model
        self.value_model = value_model
        self.device = device
        self.c_puct = float(c_puct)

        # Precompute mapping once
        self.uci_to_idx = {uci: idx for idx, uci in int_to_move.items()}
        self.policy_size = len(int_to_move)

    # --------------------------------------------------------
    # Leaf evaluation
    # --------------------------------------------------------

    def _evaluate_leaf(self, board: chess.Board) -> float:
        input_tensor = prepare_input_value_net(board).to(self.device)
        with torch.inference_mode():
            value = self.value_model(input_tensor).item()
        return float(value)

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    def _select(self, node: Node):
        while node.is_expanded():
            sqrt_N = math.sqrt(node.N) if node.N > 0 else 0.0
            best_child = None
            best_ucb = -float("inf")

            for child in node.children.values():
                q_value = 1.0 - (child.Q + 1.0) * 0.5
                ucb = q_value + self.c_puct * child.P * (sqrt_N / (1.0 + child.N))
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child

            if best_child is None:
                break

            node = best_child

        return node

    # --------------------------------------------------------
    # Expansion
    # --------------------------------------------------------

    def _expand(self, node: Node):
        input_tensor = prepare_input_policy_net(node.board).to(self.device)
        with torch.inference_mode():
            logits = self.policy_model(input_tensor).squeeze(0)
            probs = torch.softmax(logits, dim=0).cpu().numpy()

        total_prob = 0.0
        board_ref = node.board

        for move in board_ref.legal_moves:
            uci = move.uci()
            idx = self.uci_to_idx.get(uci)
            if idx is None:
                continue

            prob = float(probs[idx])

            board_ref.push(move)
            child_board = board_ref.copy(stack=False)
            board_ref.pop()

            # Create child node
            child = Node(child_board, parent=node, move=move)
            child.P = prob
            node.children[move] = child
            total_prob += prob

        # Normalize priors
        if node.children and total_prob > 0.0:
            inv = 1.0 / (total_prob + 1e-12)
            for child in node.children.values():
                child.P *= inv


    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    def _backpropagate(self, path, value):
        for node in reversed(path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def search(self, root_board: chess.Board, num_simulations: int):
        root = Node(root_board)

        for _ in range(num_simulations):
            node = root
            path = [node]

            # Selection
            node = self._select(node)
            path = self._collect_path(node)

            # Leaf
            if node.board.is_game_over():
                result = node.board.result()
                value = (
                    1.0 if result == "1-0"
                    else -1.0 if result == "0-1"
                    else 0.0
                )
            else:
                self._expand(node)
                value = self._evaluate_leaf(node.board)

            # Backprop
            self._backpropagate(path, value)

        return self._final_result(root)

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _collect_path(self, leaf):
        path = []
        node = leaf
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def _final_result(self, root: Node):
        if not root.children:
            return None, root.Q, np.zeros(self.policy_size, dtype=np.float32)

        child_items = list(root.children.items())
        visits = np.array([child.N for _, child in child_items], dtype=np.float32)
        pi = visits / (visits.sum() + 1e-12)

        policy = np.zeros(self.policy_size, dtype=np.float32)
        for (move, _), p in zip(child_items, pi):
            idx = self.uci_to_idx.get(move.uci())
            if idx is not None:
                policy[idx] = p

        best_move = max(child_items, key=lambda x: x[1].N)[0].uci()
        return best_move, root.Q, policy