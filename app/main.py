from fastapi import FastAPI
from pydantic import BaseModel
import chess
from app.ai_player import ai_select_move
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MoveRequest(BaseModel):
    fen: str


class MoveResponse(BaseModel):
    move: str


@app.post("/ai-move", response_model=MoveResponse)
def ai_move(req: MoveRequest):
    board = chess.Board(req.fen)
    move = ai_select_move(board)
    return {"move": move.uci()}