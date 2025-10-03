#!/usr/bin/env python3
import chess
import chess.engine  
from v7p3r import V7P3REngine

# Test opening simulation
print("V7P3R v12.5 Opening Simulation Test")
print("="*50)

engine = V7P3REngine()
board = chess.Board()

print(f"🤖 V7P3R v12.5 initialized")
print(f"🧠 Nudge system enabled: {hasattr(engine, 'ENABLE_NUDGE_SYSTEM') and engine.ENABLE_NUDGE_SYSTEM}")
print(f"🎯 Intelligent nudges available: {hasattr(engine, 'intelligent_nudges') and engine.intelligent_nudges is not None}")
print(f"📚 Legacy nudge database loaded: {hasattr(engine, 'nudge_database') and bool(engine.nudge_database)}")
print()

# Test first few moves
moves = []
for i in range(3):
    print(f"Move {i+1}: {board.fen()}")
    best_move = engine.search(board, time_limit=1.0)
    if best_move:
        nudge_bonus = engine._get_nudge_bonus(board, best_move)
        print(f"🎯 Selected: {best_move} (nudge: {nudge_bonus:+.1f})")
        moves.append(str(best_move))
        board.push(best_move)
    else:
        break

print()
print(f"Opening: {' '.join(moves)}")
print(f"Final: {board.fen()}")