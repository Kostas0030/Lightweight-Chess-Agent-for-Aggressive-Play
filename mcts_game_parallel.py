# CURRENT BEST
import chess
from chess import syzygy
import math
import numpy as np
import torch
from syzygy import tablebase_best_move
from auxiliary_func import prepare_input_policy_net, prepare_input_value_net


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
# Batched MCTS
# ============================================================

class MCTSBatch:
    def __init__(self, policy_model, value_model, device, int_to_move, tablebase=None, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.policy_model = policy_model
        self.value_model = value_model
        self.device = device
        self.tb = tablebase
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(dirichlet_epsilon)

        self.uci_to_idx = {uci: idx for idx, uci in int_to_move.items()}
        self.policy_size = len(int_to_move)

        self.cache = {} 
        self.max_cache_size = 50_000

    # --------------------------------------------------------
    # Selection: traverse tree to a leaf
    # --------------------------------------------------------
    def _select_path(self, root: Node):
        path = [root]
        node = root
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
            path.append(node)
        return path

    # --------------------------------------------------------
    # Expand a node using precomputed policy probabilities
    # --------------------------------------------------------
    def _expand_node(self, node: Node, probs: np.ndarray):
        total_prob = 0.0
        board_ref = node.board
        for move in board_ref.legal_moves:
            uci = move.uci()
            idx = self.uci_to_idx.get(uci)
            if idx is None:
                continue
            prob = float(probs[idx])

            if prob <= 0.0:
                continue

            board_ref.push(move)
            child_board = board_ref.copy(stack=False)
            board_ref.pop()

            child = Node(child_board, parent=node, move=move)
            child.P = prob
            node.children[move] = child
            total_prob += prob

        # Normalize priors
        if node.children and total_prob > 0.0:
            inv = 1.0 / (total_prob + 1e-12)
            for child in node.children.values():
                child.P *= inv

    def _expand_tb_move(self, node: Node, move: chess.Move):
        board_ref = node.board

        board_ref.push(move)
        child_board = board_ref.copy(stack=False)
        board_ref.pop()

        child = Node(child_board, parent=node, move=move)
        child.P = 1.0

        node.children[move] = child

    def _expand_roots_with_noise(self, roots: list):
        # --- Batch policy input ---
        policy_tensor = torch.cat(
            [prepare_input_policy_net(r.board) for r in roots],
            dim=0
        ).to(self.device)

        with torch.inference_mode():
            logits = self.policy_model(policy_tensor)

        probs_batch = torch.softmax(logits, dim=1).cpu().numpy()

        # --- Expand each root ---
        for root, probs in zip(roots, probs_batch):
            if root.is_expanded():
                continue

            self._expand_node(root, probs)

            # --- Apply Dirichlet noise ---
            '''if root.children:
                num_children = len(root.children)
                noise = np.random.dirichlet([self.dirichlet_alpha] * num_children)
                for child, n in zip(root.children.values(), noise):
                    child.P = ((1 - self.dirichlet_epsilon) * child.P + self.dirichlet_epsilon * n)'''

    # --------------------------------------------------------
    # Evaluate leaf node using value network
    # --------------------------------------------------------
    def _evaluate_tb_leaf(self, board):
        try:
            value = self.tb.probe_wdl(board)
            return float(np.clip(value / 2.0, -1.0, 1.0))
        except syzygy.MissingTableError:
            pass

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
    # Terminal board evaluation
    # --------------------------------------------------------
    def _compute_terminal_value(self, board: chess.Board):
        result = board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0

    # --------------------------------------------------------
    # Public API: search with batch support
    # --------------------------------------------------------
    def search_batch(self, root_nodes: list, num_simulations: int):
        """
        root_nodes: list of Node objects (one per game in batch)
        num_simulations: MCTS simulations per game
        """
        self._expand_roots_with_noise(root_nodes)

        for _ in range(num_simulations):
            # --- Step 1: Select expandable nodes ---
            expandable_nodes = []
            paths = []

            for root in root_nodes:
                path = self._select_path(root)
                leaf = path[-1]

                key = leaf.board._transposition_key()
                entry = self.cache.get(key)

                if leaf.board.is_game_over():
                    # Terminal node backprop
                    value = self._compute_terminal_value(leaf.board)
                    self._backpropagate(path, value)
                elif entry and "policy" in entry and "value" in entry:
                    probs = entry["policy"]
                    value = entry["value"]
                    self._expand_node(leaf, probs)
                    self._backpropagate(path, value)
                elif len(leaf.board.piece_map()) <= 5:
                    try:
                        best_move = tablebase_best_move(leaf.board, self.tb)
                    except:
                        best_move = None

                    if best_move is not None:
                        self._expand_tb_move(leaf, best_move)
                        value = self._evaluate_tb_leaf(leaf.board)
                        self._backpropagate(path, value)
                    else:
                        expandable_nodes.append(leaf)
                        paths.append(path)
                else:
                    expandable_nodes.append(leaf)
                    paths.append(path)
                    

            # --- Step 2: Batch policy and value ---
            if expandable_nodes:
                policy_tensor = torch.cat(
                    [prepare_input_policy_net(n.board) for n in expandable_nodes],
                    dim=0
                ).to(self.device)

                value_tensor = torch.cat(
                    [prepare_input_value_net(n.board) for n in expandable_nodes],
                    dim=0
                ).to(self.device)

                with torch.inference_mode():
                    logits = self.policy_model(policy_tensor)
                    values_batch = self.value_model(value_tensor).squeeze(1)
                    
                probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
                values_batch = values_batch.cpu().numpy()


                # --- Step 3: Expand nodes and backpropagate ---
                for node, probs, value, path in zip(expandable_nodes, probs_batch, values_batch, paths):
                    key = node.board._transposition_key()
                    entry = self.cache.get(key, {})
                    entry["policy"] = probs
                    entry["value"] = value
                    self.cache[key] = entry
                    
                    self._expand_node(node, probs)
                    #value = self._evaluate_leaf(node)
                    self._backpropagate(path, value)

        if len(self.cache) > self.max_cache_size:
            self.cache.clear()

    # --------------------------------------------------------
    # Extract final policy vector for a root
    # --------------------------------------------------------
    def get_pi(self, root: Node):
        if not root.children:
            return np.zeros(self.policy_size, dtype=np.float32)

        child_items = list(root.children.items())
        visits = np.array([child.N for _, child in child_items], dtype=np.float32)
        pi = visits / (visits.sum() + 1e-12)

        policy = np.zeros(self.policy_size, dtype=np.float32)
        for (move, _), p in zip(child_items, pi):
            idx = self.uci_to_idx.get(move.uci())
            if idx is not None:
                policy[idx] = p

        return policy

    # --------------------------------------------------------
    # Get best move from root
    # --------------------------------------------------------
    def get_best_move(self, root: Node):
        if not root.children:
            return None
        child_items = list(root.children.items())
        best_move = max(child_items, key=lambda x: x[1].N)[0].uci()
        return best_move