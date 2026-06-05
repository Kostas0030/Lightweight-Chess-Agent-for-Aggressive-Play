import os
import numpy as np
import chess
from chess import Board
from chess import pgn
import torch
from tqdm import tqdm
import pickle
from policy_network import PolicyNet
from value_network import ValueNet


"""
Function to load all games from a single file

input: file path of a single file
returns: all the games from that file in pgn format
"""
def load_pgn(file_path):
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games


"""
Function to load all games from all the files

input: directory path of all the games
returns: all the games from that directory in pgn format
"""
def load_games_from_directory(directory_path, limit=28):
    all_files = [file for file in os.listdir(directory_path) if file.endswith(".pgn")]
    all_games = []
    for i, file in enumerate(tqdm(all_files[:limit])):
        all_games.extend(load_pgn(os.path.join(directory_path, file)))
    print(f"GAMES PARSED: {len(all_games)} \n")
    return all_games


"""
Function that prepares the policy training data

input: games in pgn format
returns: X_tensor -> input_tensor (board state encoded), y_tensor -> target_tensor (move encoded), num_classes(around 2000) -> number of target moves, move_to_int -> Dictionary{"move" -> integer}
"""
def prepare_policy_training_data(games, max_samples=1_500_000):
    X, y = create_input_policy_net(games)
    X, y = X[:max_samples], y[:max_samples]

    y_encoded, move_to_int = encode_moves(y)
    num_classes = len(move_to_int)
    print(num_classes)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    print(f"NUMBER OF SAMPLES: {len(y_tensor)}")
    return X_tensor, y_tensor, num_classes, move_to_int


"""
Function that prepares the value training data

input: games in pgn format, number of samples
returns: X_tensor -> input_tensor (board state encoded), y_tensor -> target_tensor (result)
"""
def prepare_value_training_data(games, max_samples=5_000_000):
    X, y = create_input_value_net(games, max_samples)
    X, y = X[:max_samples], y[:max_samples]

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    print(f"NUMBER OF SAMPLES: {len(y_tensor)}")
    return X_tensor, y_tensor


"""
Function to create the input for the policy network

input: all the data games
returns: X -> matrix(N, 15, 8, 8), Y -> matrix(N, "move")
"""
def create_input_policy_net(games):
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix_policy_net(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)


"""
Function to create the input for the value network

input: all the data games, number of samples
returns: X -> matrix(N, 13, 8, 8), Y -> ,matrix(max_samples, result)
"""
def create_input_value_net(games, max_samples):
    X = []
    y = []
    for game in games:
        result_str = game.headers.get("Result")
        if result_str == "1-0":
            result = 1
        elif result_str == "0-1":
            result = -1
        else:
            result = 0

        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix_value_net(board))

            #handle side-to-move result
            if board.turn == chess.WHITE:
                y.append(result)
            else:
                y.append(-result)

            board.push(move)

            if len(X) >= max_samples:
                return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


"""
Function to convert the board into a matrix

8x8 is a size of the chess board
1-12th board for number of unique pieces
13th board for legal moves (WHERE we can move)
14th board for squares FROM WHICH we can move
15th board for side-to-move

input: board state
returns: Board -> matrix(15, 8, 8)
"""
def board_to_matrix_policy_net(board: Board):
    matrix = np.zeros((15, 8, 8))
    piece_map = board.piece_map()

    #populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    #populate the 13th channel (where pieces can move) and the 14th channel (from where pieces can move)
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        from_square = move.from_square
        row_to, col_to = divmod(to_square, 8)
        row_from, col_from = divmod(from_square, 8)
        matrix[12, row_to, col_to] = 1
        matrix[13, row_from, col_from] = 1

    #populate the 15th channel (side-to-move) with 1's if it's white-to-move and with 0's if it's black-to-move
    matrix[14] = np.ones((8, 8), dtype=np.float32) if board.turn == chess.WHITE else np.zeros((8, 8), dtype=np.float32)

    return matrix


"""
Function to convert the board into a matrix

8x8 is a size of the chess board.
1-12th board for number of unique pieces
13th board for side-to-move

input: board state
returns: Board -> matrix(13, 8, 8)
"""
def board_to_matrix_value_net(board: Board):
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    #populate the 13th channel (side-to-move) with 1's if it's white-to-move and with 0's if it's black-to-move
    matrix[12] = np.ones((8, 8), dtype=np.float32) if board.turn == chess.WHITE else np.zeros((8, 8), dtype=np.float32)

    return matrix


"""
Function to encode all the possible moves from UCI format to integer

input: UCI moves
returns: Matrix of encoded moves (integers), Dictionary{"move" -> integer}
"""
def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int


"""
Function that prepares the input for the policy network

input: board
returns: X_tensor -> tensor matrix(15, 8, 8)
"""
def prepare_input_policy_net(board: Board):
    matrix = board_to_matrix_policy_net(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor


"""
Function that prepares the input for the value network

input: board
returns: X_tensor -> tensor matrix(13, 8, 8)
"""
def prepare_input_value_net(board: Board):
    matrix = board_to_matrix_value_net(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor


# ============================================================
# Model initialization helpers (PATH-BASED)
# ============================================================

def initialize_policy_model(policy_model_path, device):
    with open("model/move_to_int", "rb") as file:
        move_to_int = pickle.load(file)

    policy_model = PolicyNet(num_classes=len(move_to_int))

    state_dict = torch.load(
        policy_model_path,
        map_location=device,
        weights_only=True
    )
    policy_model.load_state_dict(state_dict)

    policy_model.to(device)
    policy_model.eval()

    int_to_move = {v: k for k, v in move_to_int.items()}
    return policy_model, int_to_move



def initialize_value_model(value_model_path, device):
    value_model = ValueNet()

    state_dict = torch.load(
        value_model_path,
        map_location=device,
        weights_only=True
    )
    value_model.load_state_dict(state_dict)

    value_model.to(device)
    value_model.eval()

    return value_model
