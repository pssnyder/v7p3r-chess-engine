#!/usr/bin/env python3

"""
V7P3R v14.4 Tactical Analysis Debug

Debug the tactical analysis to see what patterns are being detected
and understand why move ordering isn't working as expected.
"""

import sys
import chess

# Add v7p3r engine path
v7p3r_path = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\src"
sys.path.append(v7p3r_path)

def debug_tactical_analysis():
    """Debug the tactical analysis on problematic positions"""
    
    print("V7P3R v14.4 Tactical Analysis Debug")
    print("=" * 50)
    
    try:
        from v7p3r import V7P3REngine
        engine = V7P3REngine()
        print("✅ Engine initialized")
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return
    
    # Test the problematic positions
    test_positions = [
        {
            'name': 'Fork Position (g6f4 expected)',
            'fen': 'r6k/pp4pp/3pQ1n1/2pP1rq1/2N5/8/PPP2PPP/4RRK1 b - - 3 18',
            'test_moves': ['g6f4', 'g5g2', 'g5e3', 'f5e5']
        },
        {
            'name': 'High-value Capture (f5e4 expected)',
            'fen': '3r1rk1/pp1q1p1R/2n2p1Q/4pb2/4B3/2P2NP1/PP3PP1/R3K3 b Q - 8 20',
            'test_moves': ['f5e4', 'd7d1', 'd7d2', 'f5h7']
        }
    ]
    
    for pos in test_positions:
        print(f"\n{pos['name']}")
        print("=" * 30)
        print(f"FEN: {pos['fen']}")
        
        board = chess.Board(pos['fen'])
        
        # Test each move's tactical scoring
        print(f"Move Analysis:")
        for move_str in pos['test_moves']:
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    
                    # Test the move on a copy
                    test_board = board.copy()
                    test_board.push(move)
                    
                    # Get tactical analysis
                    tactical_data = engine._analyze_position_for_tactics(test_board)
                    tactical_score = engine._calculate_tactical_move_score(board, move, tactical_data)
                    
                    # Check what type of move it is
                    move_type = "quiet"
                    if board.is_capture(move):
                        captured_piece = board.piece_at(move.to_square)
                        if captured_piece:
                            move_type = f"capture {captured_piece.symbol()}"
                    if board.gives_check(move):
                        move_type += " +check"
                    
                    print(f"  {move_str:8} ({move_type:15}) -> tactical_score: {tactical_score:4}")
                    print(f"           tactical_data: {str(tactical_data)}")
                    
            except Exception as e:
                print(f"  {move_str:8} -> ERROR: {e}")
        
        # Test full move ordering
        print(f"\nFull Move Ordering:")
        legal_moves = list(board.legal_moves)
        ordered_moves = engine._order_moves_advanced(board, legal_moves, 4)
        
        print(f"Top 10 moves:")
        for i, move in enumerate(ordered_moves[:10], 1):
            print(f"  {i:2}. {str(move)}")

if __name__ == "__main__":
    debug_tactical_analysis()