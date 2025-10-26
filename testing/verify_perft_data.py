#!/usr/bin/env python3

"""
Verify Standard Perft Test Positions

Let's double-check the known perft test positions to ensure our test data is correct.
"""

import chess

def verify_perft_positions():
    """Verify the standard perft test positions"""
    
    # Standard perft test positions from the Chess Programming Wiki
    positions = [
        {
            "name": "Position 1 (Start)",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "depths": {
                1: 20,
                2: 400,
                3: 8902,
                4: 197281
            }
        },
        {
            "name": "Position 2 (Kiwipete)",
            "fen": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
            "depths": {
                1: 48,
                2: 2039,
                3: 97862
            }
        },
        {
            "name": "Position 3",
            "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
            "depths": {
                1: 14,
                2: 191,
                3: 2812,
                4: 43238
            }
        },
        {
            "name": "Position 4",
            "fen": "r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1",
            "depths": {
                1: 8,  # This was our test data - let's verify
                2: 239,
                3: 2812
            }
        },
        {
            "name": "Position 5",
            "fen": "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "depths": {
                1: 44,
                2: 1486,
                3: 62379
            }
        }
    ]
    
    print("Verifying perft test positions with python-chess:")
    print("=" * 60)
    
    for pos in positions:
        print(f"\n{pos['name']}")
        print(f"FEN: {pos['fen']}")
        
        board = chess.Board(pos['fen'])
        actual_moves = len(list(board.legal_moves))
        expected_depth1 = pos['depths'].get(1, "Unknown")
        
        print(f"Expected depth 1: {expected_depth1}")
        print(f"Actual depth 1:   {actual_moves}")
        
        if expected_depth1 != "Unknown":
            if actual_moves == expected_depth1:
                print("✅ MATCH")
            else:
                print("❌ MISMATCH - Test data may be wrong!")
                print(f"   Difference: {actual_moves - expected_depth1}")

if __name__ == "__main__":
    verify_perft_positions()