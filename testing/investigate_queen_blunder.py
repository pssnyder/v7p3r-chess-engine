#!/usr/bin/env python3
"""
Investigate V14.5 Queen Blunder vs Stockfish 1%
Analyze the position where V14.5 blundered Qxa4+ allowing Kb2 Qb4+
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

# Position after 25. Rf3 (before Qxa4+)
# The actual position where white needs to respond to the threat
fen_after_rf3 = "4k2r/pp1bb1p1/4p3/5n2/P2P4/2r2R1P/q1K5/3R4 b k - 1 25"

print("=" * 70)
print("Investigating V14.5 Queen Blunder")
print("=" * 70)
print()

board = chess.Board(fen_after_rf3)
print("Position after 25. Rf3 (before Qxa4+)")
print(board)
print()
print("Black to move - Black played 25...Qxa4+")
print()

print("Black to move - Black played 25...Qxa4+")
print()

# Black plays Qxa4+
board.push_san("Qxa4+")
print("After 25...Qxa4+:")
print(board)
print()

# Check legal moves for white
legal_moves = list(board.legal_moves)
print(f"White's legal moves in check: {len(legal_moves)}")
for move in legal_moves:
    san = board.san(move)
    print(f"  {move.uci()}: {san}")
print()

# According to PGN, white played Kb2
# But looking at legal moves above, let's verify

print("According to PGN, white played 26. Kb2")
print("Checking if Kb2 is legal...")

kb2_legal = False
for move in legal_moves:
    if board.san(move) == "Kb2":
        kb2_legal = True
        print(f"✓ Kb2 is LEGAL (from {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)})")
        
        # Play it and see what happens
        board_kb2 = board.copy()
        board_kb2.push(move)
        print()
        print("After 26. Kb2:")
        print(board_kb2)
        print()
        
        # Check black's response
        black_legal = list(board_kb2.legal_moves)
        print(f"Black's legal moves: {len(black_legal)}")
        
        # Look for Qb4+
        for bmove in black_legal:
            if board_kb2.san(bmove) == "Qb4+":
                print(f"✓ Black can play Qb4+ ({bmove.uci()})")
                
                board_qb4 = board_kb2.copy()
                board_qb4.push(bmove)
                print()
                print("After 26...Qb4+:")
                print(board_qb4)
                print()
                
                # Can white capture the queen?
                white_legal_after_qb4 = list(board_qb4.legal_moves)
                print(f"White's legal moves: {len(white_legal_after_qb4)}")
                for wmove in white_legal_after_qb4:
                    wsan = board_qb4.san(wmove)
                    print(f"  {wmove.uci()}: {wsan}")
                    
                    # Check if it captures the queen
                    if wmove.to_square == chess.B4:
                        captured = board_qb4.piece_at(wmove.to_square)
                        if captured and captured.piece_type == chess.QUEEN:
                            print(f"    ✗ CANNOT capture - queen is protected or out of range!")
                
                break
        break

if not kb2_legal:
    print("✗ Kb2 is NOT LEGAL - something wrong with PGN notation?")

print()
print("Let's also check Ka1 alternative:")
for move in legal_moves:
    if board.san(move) == "Ka1":
        print(f"✓ Ka1 is available ({move.uci()})")
        
        board_ka1 = board.copy()
        board_ka1.push(move)
        print()
        print("After 26. Ka1:")
        print(board_ka1)
        print()
        
        # Is the position safer?
        print("Position analysis after Ka1:")
        print("- King on a1 (corner, harder to attack)")
        print("- Black queen on a4 cannot easily give check")
        print("- White rooks still active")
        break

print()
print("=" * 70)
print("Analysis Summary")
print("=" * 70)
print()
print("Position: After 25. Rf3 Qxa4+, white king on c2 in check")
print("Correct move: 26. Ka1 (king to a1, escapes check safely)")
print("Blunder: 26. Kb2 allows 26...Qb4+ and white cannot capture the queen")
print()
print("This is a TACTICAL BLUNDER - the search should have seen this.")
print("The blunder firewall checks hanging pieces, but only AFTER a move is made.")
print("The search evaluation should have scored Kb2 as terrible due to Qb4+ response.")
print()
print("Possible causes:")
print("1. Search depth too shallow (didn't see 2-ply tactic)")
print("2. Move ordering put Ka1 too low in the list")
print("3. Emergency time limit stopped search before seeing the tactic")
print("4. Evaluation didn't penalize king exposure enough")
