import chess
MOVE_TO_IDX = {}
IDX_TO_MOVE = {}

def build_action_space():
    idx = 0
    board = chess.Board()
    squares = list(chess.SQUARES)
    promotions = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

    for from_sq in squares:
        for to_sq in squares:
            for promo in promotions:
                try:
                    move = chess.Move(from_sq, to_sq, promotion=promo)
                    if move.uci() not in MOVE_TO_IDX:
                        MOVE_TO_IDX[move.uci()] = idx
                        IDX_TO_MOVE[idx] = move.uci()
                        idx += 1
                except:
                    pass

    print("Total actions:", idx)

def encode_move(move: chess.Move):
    return MOVE_TO_IDX[move.uci()]

def decode_move(idx):
    return chess.Move.from_uci(IDX_TO_MOVE[idx])