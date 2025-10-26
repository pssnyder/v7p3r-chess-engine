#!/usr/bin/env python3

"""
Investigate Position 4 Discrepancy

There's something wrong with Position 4. Let me carefully check the FEN.
"""

import chess

def investigate_position_4():
    """Investigate the Position 4 discrepancy"""
    
    # From the wiki, Position 4 should be:
    suspected_fen = "r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1"
    
    print("Investigating Position 4:")
    print(f"FEN: {suspected_fen}")
    print()
    
    try:
        board = chess.Board(suspected_fen)
        print("Board visualization:")
        print(board)
        print()
        
        legal_moves = list(board.legal_moves)
        print(f"Total legal moves: {len(legal_moves)}")
        print("Legal moves:")
        for i, move in enumerate(legal_moves, 1):
            print(f"{i:2d}. {move}")
        print()
        
        # Let's check if there are any problems with the position
        print("Position analysis:")
        print(f"Turn: {'Black' if board.turn == chess.BLACK else 'White'}")
        print(f"Castling rights: {board.castling_rights}")
        print(f"En passant: {board.ep_square}")
        print(f"Halfmove clock: {board.halfmove_clock}")
        print(f"Fullmove number: {board.fullmove_number}")
        
        # Check if this position is legal
        print(f"Position is legal: {board.is_valid()}")
        
    except Exception as e:
        print(f"Error with FEN: {e}")

    # Let me also try some variations that might be correct
    print("\n" + "="*50)
    print("Let me try some alternative Position 4 FENs...")
    
    # Maybe it's this one (from some other source)?
    alternatives = [
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
    ]
    
    for i, fen in enumerate(alternatives, 1):
        print(f"\nAlternative {i}: {fen}")
        try:
            board = chess.Board(fen)
            moves = len(list(board.legal_moves))
            print(f"Legal moves: {moves}")
            if moves == 6:
                print("✅ This might be the correct Position 4!")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    investigate_position_4()