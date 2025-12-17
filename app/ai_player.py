import torch
import chess
import random
from app.model import ChessNet
from app.board_utils import board_to_tensor
from app.action_utils import encode_move, IDX_TO_MOVE, build_action_space

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

build_action_space()

model = ChessNet().to(DEVICE)
try:
    model.load_state_dict(
        torch.load("models/latest_model.pt", map_location=DEVICE)
    )
    model.eval()
except Exception:
    model = None

def ai_select_move(board: chess.Board, temperature: float = 1.0):
    if model is None:
        return random.choice(list(board.legal_moves))
    state = torch.tensor(
        board_to_tensor(board),
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(state)
    mask = torch.full_like(logits, -1e9)
    for move in board.legal_moves:
        mask[0, encode_move(move)] = 0.0
    masked_logits = logits + mask
    probs = torch.softmax(masked_logits / temperature, dim=1)
    move_idx = torch.multinomial(probs[0], 1).item()
    return chess.Move.from_uci(IDX_TO_MOVE[move_idx])
