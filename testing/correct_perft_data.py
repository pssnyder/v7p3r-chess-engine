#!/usr/bin/env python3

"""
Fix Perft Test Data

The Position 4 from the chess programming wiki is different than what we used.
Let's get the correct perft test positions.
"""

import chess

def get_correct_perft_positions():
    """Get the actual standard perft test positions"""
    
    # These are the CORRECT standard perft test positions
    positions = [
        {
            "name": "Position 1 (Initial)",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "depths": {1: 20, 2: 400, 3: 8902, 4: 197281, 5: 4865609}
        },
        {
            "name": "Position 2 (Kiwipete)",
            "fen": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
            "depths": {1: 48, 2: 2039, 3: 97862, 4: 4085603}
        },
        {
            "name": "Position 3",
            "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
            "depths": {1: 14, 2: 191, 3: 2812, 4: 43238, 5: 674624}
        },
        {
            "name": "Position 4 (CORRECT from wiki)",
            "fen": "r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1",
            "depths": {1: 6, 2: 264, 3: 9467, 4: 422333}
        },
        {
            "name": "Position 5",
            "fen": "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "depths": {1: 44, 2: 1486, 3: 62379}
        }
    ]
    
    print("CORRECT Perft Test Positions:")
    print("=" * 60)
    
    for pos in positions:
        print(f"\n{pos['name']}")
        print(f"FEN: {pos['fen']}")
        
        board = chess.Board(pos['fen'])
        actual_moves = len(list(board.legal_moves))
        expected_depth1 = pos['depths'][1]
        
        print(f"Expected depth 1: {expected_depth1}")
        print(f"Actual depth 1:   {actual_moves}")
        
        if actual_moves == expected_depth1:
            print("✅ CORRECT")
        else:
            print(f"❌ STILL WRONG - Python-chess shows {actual_moves}")

if __name__ == "__main__":
    get_correct_perft_positions()