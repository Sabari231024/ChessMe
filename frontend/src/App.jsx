import React, { useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";

export default function App() {
    const gameRef = useRef(new Chess());
    const [fen, setFen] = useState(gameRef.current.fen());

    console.log("Render App, fen =", fen);

    function onPieceDrop({ sourceSquare, targetSquare, piece }) {
        console.log("DROP EVENT FIRED");
        console.log("from:", sourceSquare, "to:", targetSquare, "piece:", piece);

        const move = gameRef.current.move({
            from: sourceSquare,
            to: targetSquare,
            promotion: "q"
        });

        console.log("chess.js move result:", move);

        if (move === null) {
            console.log("Illegal move");
            return false;
        }

        setFen(gameRef.current.fen());
        return true;
    }

    return (
        <div
            style={{
                minHeight: "100vh",
                background: "#1e1e1e",
                display: "flex",
                justifyContent: "center",
                alignItems: "center"
            }}
        >
            <Chessboard
                id="DebugBoard"
                boardWidth={420}
                position={fen}
                onPieceDrop={onPieceDrop}
            />
        </div>
    );
}
