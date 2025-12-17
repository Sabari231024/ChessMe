import torch
import torch.nn.functional as F
from app.model import ChessNet
from app.memory import load_last_game
from app.action_utils import encode_move, build_action_space

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_on_last_game(epochs=20, lr=1e-4):
    build_action_space()
    episode = load_last_game()
    if episode is None:
        print("No game to train on")
        return


    model = ChessNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load("models/latest_model.pt"))
    except:
        print("Training from scratch")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    z = torch.tensor([episode["result"]], device=DEVICE)

    for epoch in range(epochs):
        total_loss = 0
        for state, move_uci in zip(episode["states"], episode["actions"]):
            s = torch.tensor(state).unsqueeze(0).to(DEVICE)
            a = encode_move(chess.Move.from_uci(move_uci))

            policy_logits, value = model(s)
            policy_loss = F.cross_entropy(policy_logits, torch.tensor([a], device=DEVICE))
            value_loss = (value.squeeze() - z) ** 2

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()


        print(f"Epoch {epoch+1}: loss={total_loss:.3f}")


torch.save(model.state_dict(), "models/latest_model.pt")
print("Model updated and saved")