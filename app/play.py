import chess
from app.board_utils import board_to_tensor
from app.memory import save_last_game
from app.ai_player import ai_select_move

board = chess.Board()
states, actions = [], []

while not board.is_game_over():
    print(board)

    move = chess.Move.from_uci(input("Your move: "))
    if move not in board.legal_moves:
        print("Illegal move")
        continue

    states.append(board_to_tensor(board))
    actions.append(move.uci())
    board.push(move)

    if board.is_game_over():
        break

    ai_move = ai_select_move(board)
    states.append(board_to_tensor(board))
    actions.append(ai_move.uci())
    board.push(ai_move)

result = board.result()
z = 1 if result == "1-0" else -1 if result == "0-1" else 0

save_last_game({
"states": states,
"actions": actions,
"result": z
})


print("Game stored. Result:", result)