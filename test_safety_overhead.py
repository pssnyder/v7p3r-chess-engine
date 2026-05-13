import chess
import time
from v7p3r_move_safety import MoveSafetyChecker

board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
safety = MoveSafetyChecker({chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000})

# Test safety check overhead
moves = list(board.legal_moves)
start = time.time()
for _ in range(1000):
    for move in moves:
        safety.evaluate_move_safety(board, move)
elapsed = time.time() - start
checks_per_sec = (1000 * len(moves)) / elapsed
print(f"Safety checks/sec: {int(checks_per_sec)}")
print(f"Time per check: {elapsed/(1000*len(moves))*1000:.3f}ms")
