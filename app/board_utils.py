import chess
import numpy as np


PIECE_MAP = {
chess.PAWN: 0,
chess.KNIGHT: 1,
chess.BISHOP: 2,
chess.ROOK: 3,
chess.QUEEN: 4,
chess.KING: 5,
}




def board_to_tensor(board: chess.Board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for piece_type in PIECE_MAP:
        for square in board.pieces(piece_type, chess.WHITE):
            r, c = divmod(square, 8)
            tensor[PIECE_MAP[piece_type], r, c] = 1
        for square in board.pieces(piece_type, chess.BLACK):
            r, c = divmod(square, 8)
            tensor[PIECE_MAP[piece_type] + 6, r, c] = 1
    return tensor