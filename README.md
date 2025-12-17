# ♟️ ChessMe — Continual Learning Chess AI

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat&logo=react&logoColor=black)](https://reactjs.org)

> **A self-improving chess agent that learns from every game you play against it.**

ChessMe is an Actor-Critic reinforcement learning chess engine that continuously improves through gameplay. Unlike traditional chess engines that rely on pre-computed knowledge, ChessMe learns from your moves and its own mistakes — becoming stronger with each session.

---

## 🎯 Project Vision

Build a chess AI that:
- **Learns like a human** — improves through post-game analysis
- **Remembers your style** — adapts to how you play
- **Never stops growing** — every game makes it stronger

---

## 🧠 How It Works

### Actor-Critic Architecture

The heart of ChessMe is an **Actor-Critic neural network** — a powerful reinforcement learning paradigm that combines two complementary learning signals:

```
┌─────────────────────────────────────────────────────────────┐
│                    ChessNet Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│     Input: 12×8×8 Board Tensor (piece positions)            │
│                        │                                    │
│                        ▼                                    │
│     ┌───────────────────────────────────┐                   │
│     │   Convolutional Feature Extractor │                   │
│     │   Conv2d(12→64) → ReLU            │                   │
│     │   Conv2d(64→128) → ReLU           │                   │
│     └───────────────────────────────────┘                   │
│                        │                                    │
│                        ▼                                    │
│              Flatten: 128×8×8 = 8192                        │
│                   /         \                               │
│                  /           \                              │
│                 ▼             ▼                             │
│     ┌─────────────────┐ ┌─────────────────┐                 │
│     │   Policy Head   │ │   Value Head    │                 │
│     │   (Actor)       │ │   (Critic)      │                 │
│     │                 │ │                 │                 │
│     │ Linear(8192→    │ │ Linear(8192→1)  │                 │
│     │        4672)    │ │      ↓          │                 │
│     │      ↓          │ │    tanh()       │                 │
│     │  Softmax        │ │                 │                 │
│     └─────────────────┘ └─────────────────┘                 │
│            │                    │                           │
│            ▼                    ▼                           │
│     Move Probabilities    Position Value                    │
│     (4672 legal moves)    (-1 to +1)                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

| Component | Purpose |
|-----------|---------|
| **Actor (Policy Head)** | Outputs probability distribution over all possible moves |
| **Critic (Value Head)** | Estimates how good the current position is (win probability) |

### Why Actor-Critic?

| Approach | Limitation |
|----------|------------|
| Policy-only (REINFORCE) | High variance, unstable learning |
| Value-only (DQN) | Can't handle continuous/large action spaces well |
| **Actor-Critic** | Best of both — stable, efficient, scalable ✓ |

---

## 🔄 Continual Learning Loop

```
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    ▼                                                 │
┌─────────┐    ┌─────────────┐    ┌───────────────┐   │
│ Play a  │───▶│ Store Game  │───▶│ Train Model   │───┘
│ Game    │    │ (memory.py) │    │ (train.py)    │
└─────────┘    └─────────────┘    └───────────────┘
                     │                    │
                     │                    ▼
                     │          ┌─────────────────┐
                     │          │ Next Game:      │
                     │          │ Stronger AI!    │
                     │          └─────────────────┘
                     │
                     ▼
            Game data includes:
            • All board states
            • All moves made
            • Final result (+1/-1)
```

### Learning Process

1. **You play a game** against the AI (as white or black)
2. **Game is recorded** — every position, every move, final outcome
3. **Post-game training** — the model replays the game multiple times:
   - **Policy head** learns: "In position X, move Y won/lost the game"
   - **Value head** learns: "Position X led to a win/loss"
4. **Next session** — the AI plays with updated knowledge

> 💡 *This is how humans improve: analyze your games, learn from mistakes, do better next time.*

---

## ✨ Key Features

### ✅ Currently Implemented

| Feature | Description |
|---------|-------------|
| **Legal Move Masking** | AI only considers valid chess moves — no probability wasted on illegal moves |
| **Temperature-Based Selection** | Controls exploration vs exploitation (higher = more varied play) |
| **Game Memory System** | Persists game data between sessions for continual learning |
| **Full Action Space** | Supports all 4,672 possible moves including promotions |
| **GPU Acceleration** | CUDA support for fast training and inference |
| **Web Interface** | Beautiful React chessboard with drag-and-drop |
| **FastAPI Backend** | Production-ready API for AI moves |

### 🚧 Phase 3 — Coming Next

| Feature | Benefit |
|---------|---------|
| **Entropy Regularization** | Prevents overfitting to specific games |
| **Enhanced Temperature Scheduling** | Smarter exploration during training |
| **Self-Play Mode** | AI plays against itself for accelerated learning |

---

## 📁 Project Structure

```
ChessMe/
├── app/                          # Backend (Python)
│   ├── model.py                  # ChessNet: Actor-Critic neural network
│   ├── ai_player.py              # Move selection with legal masking
│   ├── train.py                  # Training loop (learns from games)
│   ├── memory.py                 # Game persistence (pickle storage)
│   ├── board_utils.py            # Board → Tensor conversion
│   ├── action_utils.py           # Move encoding/decoding (4672 actions)
│   ├── main.py                   # FastAPI server
│   └── play.py                   # CLI play mode
│
├── frontend/                     # Frontend (React + Vite)
│   ├── src/
│   │   └── App.jsx               # Chess UI with react-chessboard
│   ├── package.json
│   └── vite.config.js
│
├── models/                       # Saved model weights
│   └── latest_model.pt           # Current best model
│
├── data/                         # Game storage
│   └── last_game.pkl             # Most recent game for training
│
└── requirements.txt              # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- CUDA-capable GPU (optional, but recommended)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ChessMe.git
cd ChessMe

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch python-chess fastapi uvicorn pydantic numpy

# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
# In a new terminal
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Visit **http://localhost:5173** to play!

---

## 🎮 How to Play

1. **Start both servers** (backend on :8000, frontend on :5173)
2. **Open the web UI** in your browser
3. **Play as White** — drag and drop pieces
4. **AI responds** — the model calculates and plays its move
5. **After the game** — run training to improve the AI:

```bash
python -m app.train
```

---

## 🔧 Configuration

### Temperature (in `ai_player.py`)

```python
ai_select_move(board, temperature=1.0)
```

| Value | Behavior |
|-------|----------|
| `0.1` | Deterministic — always plays "best" move |
| `1.0` | Balanced — probabilistic selection |
| `2.0` | Exploratory — more random, creative moves |

### Training Parameters (in `train.py`)

```python
train_on_last_game(epochs=20, lr=1e-4)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 20 | Times to replay the game |
| `lr` | 1e-4 | Learning rate (lower = more stable) |

---

## 📊 Technical Details

### Board Representation

The board is encoded as a **12×8×8 tensor**:
- 6 channels for White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
- 6 channels for Black pieces
- Binary encoding: 1 = piece present, 0 = empty

### Action Space

| Moves | Description |
|-------|-------------|
| 4,672 | All possible (from_square, to_square, promotion) combinations |

Legal move masking ensures invalid moves get probability ≈ 0.

### Loss Function

```
Total Loss = Policy Loss + 0.5 × Value Loss

Policy Loss = CrossEntropy(predicted_move, actual_move)
Value Loss  = (predicted_value - game_result)²
```

---

## 🧪 Training Tips

1. **Play lots of games** — more data = better learning
2. **Complete games** — resigned games still provide valuable signal
3. **Mix play styles** — occasionally explore unusual moves
4. **Train incrementally** — run training after every few games

---

## 🛣️ Roadmap

### Phase 1 ✅ — Foundation
- [x] Actor-Critic network architecture
- [x] Board state encoding
- [x] Legal move masking
- [x] Game memory system
- [x] FastAPI + React integration

### Phase 2 ✅ — Core Learning
- [x] Training from recorded games
- [x] Temperature-based move selection
- [x] Model persistence
- [x] Gradient clipping for stability

### Phase 3 🔜 — Enhancement
- [ ] Entropy regularization
- [ ] Advanced temperature scheduling
- [ ] Opening book integration
- [ ] Evaluation mode (deterministic play)

### Phase 4 🔮 — Advanced
- [ ] Self-play training
- [ ] Experience replay buffer
- [ ] Multi-game batch training
- [ ] ELO tracking system

---

## 🧬 Why This Approach Works

Traditional chess engines (like Stockfish) use:
- Hand-crafted evaluation functions
- Alpha-beta pruning
- Massive opening books

**ChessMe is different:**
- Learns from scratch through gameplay
- Discovers patterns naturally
- Adapts to opponent styles
- Gets better over time — not static

> This is the same core idea behind **AlphaGo/AlphaZero**, simplified for personal use.

---

## 📚 References

- [Actor-Critic Methods](https://arxiv.org/abs/1602.01783) — Mnih et al.
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) — Silver et al.
- [python-chess Library](https://python-chess.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## 🤝 Contributing

Contributions are welcome! Areas that need work:
- Improved board evaluation
- Self-play infrastructure
- Mobile-friendly UI
- Game analysis tools

---

## 📄 License

MIT License — feel free to use, modify, and share.

---

<div align="center">

**Made with ♟️ and 🧠 by the ChessMe team**

*Teach your AI to play chess — one game at a time.*

</div>

ReadMe File created by Antigravity AI
