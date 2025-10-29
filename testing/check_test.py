#!/usr/bin/env python3

"""
Check if the top moves are actually checks
"""

import sys
import chess

def test_check_moves():
    # Fork position
    board = chess.Board('r6k/pp4pp/3pQ1n1/2pP1rq1/2N5/8/PPP2PPP/4RRK1 b - - 3 18')
    
    test_moves = ['g5g2', 'g5e3', 'g6f4']
    
    for move_str in test_moves:
        move = chess.Move.from_uci(move_str)
        is_check = board.gives_check(move)
        print(f"{move_str}: Check = {is_check}")

if __name__ == "__main__":
    test_check_moves()