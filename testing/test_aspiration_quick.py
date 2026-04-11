"""
Quick Aspiration Windows Baseline Test
Fast 10-position test to establish baseline node counts.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

# 10 representative positions
test_positions = [
    ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4", "Petrov Defense"),
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5", "Italian middlegame"),
    ("r1bq1rk1/ppp2ppp/2npbn2/4p3/2BPP3/2N2N2/PPP2PPP/R1BQ1RK1 b - - 0 8", "Closed center"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4", "Fork threat"),
    ("r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9", "Symmetric middlegame"),
    ("8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 w - - 0 1", "Pawn endgame"),
    ("6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1", "Rook endgame"),
    ("r1bq1rk1/pp1nbppp/2p2n2/3pp3/2PP4/2NBPN2/PP2BPPP/R1BQK2R w KQ - 2 9", "Queenside expansion"),
    ("rnbqk2r/pp2bppp/2p2n2/3p4/3PP3/2N2N2/PP3PPP/R1BQKB1R w KQkq - 0 7", "French Tarrasch"),
    ("r2qkb1r/pp1n1ppp/2pbpn2/3p4/2PP4/2NBPN2/PP2BPPP/R2QK2R w KQkq - 2 8", "Semi-open file"),
]

engine = V7P3REngine()
total_nodes = 0

print("\n=== Quick Baseline: 10 Positions at Depth 4 ===\n")

for i, (fen, description) in enumerate(test_positions, 1):
    board = chess.Board(fen)
    engine.nodes_searched = 0
    
    best_move = engine.search(board, depth=4)
    nodes = engine.nodes_searched
    total_nodes += nodes
    
    print(f"{i:2d}. {nodes:6d} nodes - {description}")

avg_nodes = total_nodes / len(test_positions)
print(f"\nAverage Nodes: {int(avg_nodes)}")
print(f"Total Nodes: {total_nodes}")
print("\n✅ BASELINE ESTABLISHED")
print(f"Expected after aspiration: ~{int(avg_nodes * 0.80)} avg nodes (20% reduction)")
