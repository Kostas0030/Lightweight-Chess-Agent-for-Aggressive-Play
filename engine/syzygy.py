import chess
from chess import Board
from chess import syzygy


def tablebase_best_move(board, tb):
    try:
        root_wdl = tb.probe_wdl(board)
        root_dtz = tb.probe_dtz(board)
    except syzygy.MissingTableError:
        return None

    best_move = None
    best_score = None

    for move in board.legal_moves:
        new_board = board.copy()
        new_board.push(move)

        try:
            wdl = tb.probe_wdl(new_board)
            dtz = tb.probe_dtz(new_board)
        except syzygy.MissingTableError:
            continue

        if new_board.is_game_over() and -wdl == root_wdl and root_dtz == 1:
            #print("Legal move:", move, ", Other-side WDL:", wdl, ", Other-side DTZ:", dtz, ", Score:", score, ", Best score:", best_score)
            best_move = move
            return move
        
        is_progress = board.is_capture(move) or board.piece_type_at(move.from_square) == chess.PAWN
        if (root_wdl >= 0 and -wdl == root_wdl and -dtz < root_dtz) or (root_wdl >= 0 and -wdl == root_wdl and is_progress): #if winning, keep winning state (wdl) the same, and minimize distance to zero (dtz)
            if root_wdl == 2:
                score = dtz
            if is_progress:
                 score = 10000
            else:
                score = 0 #in case of a cursed win (50-move rule), make all the moves look the same (draw anyways)
        elif root_wdl >= 0 and -wdl == root_wdl and -dtz >= root_dtz:
            if root_wdl == 2:
                score = dtz
            else:
                score = 0 #in case of a cursed win (50-move rule), make all the moves look the same (draw anyways)
        elif root_wdl <= 0 and dtz < -root_dtz: #if losing, can't escape loss, but maximize distance to zero (dtz)
            if root_wdl == -2:
                score = dtz
            else:
                score = 0 #in case of a blessed loss (50-move rule), make all the moves look the same (draw anyways)
        else:
            score = -9999 #make all the moves that don't preserve the winning state look bad
        #print("Legal move:", move, ", Other-side WDL:", wdl, ", Other-side DTZ:", dtz, ", Score:", score, ", Best score:", best_score)
        if best_score is None or score > best_score:
            best_score = score
            best_move = move

    return best_move



def main():
    tb = syzygy.open_tablebase("data/tablebase")

    board = Board("8/8/3k4/8/7r/4K2p/7P/8 w - - 22 12")  #Black to move

    while not board.is_game_over():
        print("\n")
        wdl = tb.probe_wdl(board)
        dtz = tb.probe_dtz(board)
        print("WDL =", wdl, "DTZ =", dtz)
        print(board)
        print(board.fen())
        best_move = tablebase_best_move(board, tb)
        print("Returned:", best_move)
        board.push(best_move)



if __name__ == "__main__":
    main()


