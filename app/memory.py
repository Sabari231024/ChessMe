import pickle
from pathlib import Path

DATA_PATH = Path("data/last_game.pkl")

def save_last_game(episode):
    DATA_PATH.parent.mkdir(exist_ok=True)
    with open(DATA_PATH, "wb") as f:
        pickle.dump(episode, f)

def load_last_game():
    if not DATA_PATH.exists():
        return None
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)