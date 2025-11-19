#!/usr/bin/env python3
"""
V15.2 Self-Play Consistency Test
Tests V15.2 playing against itself to check for perspective consistency.
If the perspective bug is fixed, V15.2 should play sensibly as both colors.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import chess.pgn
import time
from datetime import datetime
from v7p3r import V7P3REngine


class TimedEngine:
    def __init__(self, name: str, time_per_move: float = 6.0):
        self.name = name
        self.engine = V7P3REngine()
        self.time_per_move = time_per_move
        
    def get_move(self, board: chess.Board) -> chess.Move:
        """Get move from engine with time limit"""
        start_time = time.time()
        move = self.engine.get_best_move(time_left=self.time_per_move, increment=0.1)
        elapsed = time.time() - start_time
        
        if move in board.legal_moves:
            print(f"  {self.name} -> {move} ({elapsed:.1f}s)")
            return move
        else:
            # Fallback to first legal move if engine returns illegal move
            legal_moves = list(board.legal_moves)
            fallback_move = legal_moves[0] if legal_moves else None
            print(f"  {self.name} -> ILLEGAL {move}, using fallback {fallback_move}")
            return fallback_move


def play_position_test(starting_fen: str, test_name: str, max_moves: int = 20) -> dict:
    """Play from a specific position to test consistency"""
    print(f"\n{'='*50}")
    print(f"POSITION TEST: {test_name}")
    print(f"{'='*50}")
    
    # Create two instances (representing same engine as different colors)
    white_engine = TimedEngine("V15.2-White", 6.0)
    black_engine = TimedEngine("V15.2-Black", 6.0)
    
    board = chess.Board(starting_fen)
    print(f"Starting position: {starting_fen}")
    print(board)
    print()
    
    moves = []
    move_count = 0
    
    while not board.is_game_over() and move_count < max_moves:
        current_engine = white_engine if board.turn == chess.WHITE else black_engine
        
        move_count += 1
        print(f"Move {(move_count + 1) // 2}: {current_engine.name}")
        
        try:
            move = current_engine.get_move(board)
            if move is None:
                print("  No legal moves available")
                break
                
            moves.append(move.uci())
            board.push(move)
            
            # Show key moves
            if move_count <= 6:
                print(f"    Position: {board.fen()}")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            break
    
    # Analyze the game
    result = "1/2-1/2"  # Default to draw
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        result = "0-1" if board.turn == chess.WHITE else "1-0"
        print(f"\nCHECKMATE! Winner: {winner}")
    elif board.is_stalemate():
        print(f"\nSTALEMATE!")
    elif board.is_insufficient_material():
        print(f"\nDRAW: Insufficient material")
    else:
        print(f"\nGAME STOPPED after {move_count} moves")
    
    print(f"Final position ({move_count} moves): {board.fen()}")
    print(f"Move sequence: {' '.join(moves[:10])}{'...' if len(moves) > 10 else ''}")
    
    return {
        'result': result,
        'moves': moves,
        'final_fen': board.fen(),
        'move_count': move_count
    }


def analyze_opening_consistency():
    """Test V15.2's opening play as White and Black"""
    print("V7P3R v15.2 - Opening Consistency Test")
    print("Testing for perspective bug via opening move patterns")
    print()
    
    # Test 1: Starting position - how does V15.2 open as White?
    white_test = play_position_test(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Opening as White",
        10
    )
    
    # Test 2: After 1.e4 - how does V15.2 respond as Black?
    black_test = play_position_test(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "Response to 1.e4 as Black", 
        10
    )
    
    # Test 3: After 1.d4 - how does V15.2 respond as Black?
    black_test2 = play_position_test(
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
        "Response to 1.d4 as Black",
        10
    )
    
    return [white_test, black_test, black_test2]


def test_middlegame_tactics():
    """Test tactical consistency in middlegame positions"""
    print(f"\n{'='*60}")
    print("MIDDLEGAME TACTICAL TESTS")
    print(f"{'='*60}")
    
    # Test tactical position where material can be won
    tactical_test = play_position_test(
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
        "Italian Game - Tactical Position",
        12
    )
    
    # Test endgame position
    endgame_test = play_position_test(
        "8/8/8/8/8/3k4/3P4/3K4 w - - 0 1",
        "King and Pawn vs King Endgame",
        15
    )
    
    return [tactical_test, endgame_test]


def main():
    """Run comprehensive consistency tests"""
    print("="*60)
    print("V7P3R v15.2 - PERSPECTIVE BUG VALIDATION")
    print("Testing for consistent play as White and Black")
    print("="*60)
    
    all_tests = []
    
    # Opening tests
    opening_results = analyze_opening_consistency()
    all_tests.extend(opening_results)
    
    # Middlegame tests  
    middlegame_results = test_middlegame_tactics()
    all_tests.extend(middlegame_results)
    
    # Analysis
    print(f"\n{'='*60}")
    print("PERSPECTIVE CONSISTENCY ANALYSIS")
    print(f"{'='*60}")
    
    print("Key Indicators of Fixed Perspective Bug:")
    print("✓ Both colors make logical developing moves")
    print("✓ No obvious 'anti-moves' (edge pawns, decentralizing pieces)")
    print("✓ Similar quality of play regardless of color")
    print()
    
    # Check for common anti-patterns that indicated the bug
    suspicious_moves = []
    good_moves = []
    
    for i, test in enumerate(all_tests):
        print(f"Test {i+1}: {len(test['moves'])} moves, result: {test['result']}")
        
        # Look for edge pawn moves (h3, h6, a4, a5, etc.) early in game
        early_moves = test['moves'][:6]  # First 6 moves
        edge_moves = [m for m in early_moves if m in ['h2h3', 'h2h4', 'a2a3', 'a2a4', 'h7h6', 'h7h5', 'a7a6', 'a7a5']]
        
        # Look for central moves (good signs)
        central_moves = [m for m in early_moves if m in ['e2e4', 'd2d4', 'e7e5', 'd7d5', 'g1f3', 'b1c3', 'g8f6', 'b8c6']]
        
        if edge_moves:
            print(f"  ⚠️  Edge moves detected: {edge_moves}")
            suspicious_moves.extend(edge_moves)
        
        if central_moves:
            print(f"  ✓ Central/developing moves: {central_moves}")
            good_moves.extend(central_moves)
    
    print(f"\nSUMMARY:")
    print(f"Suspicious edge moves: {len(suspicious_moves)}")
    print(f"Good central/developing moves: {len(good_moves)}")
    
    if len(good_moves) > len(suspicious_moves) * 2:
        print("✅ PERSPECTIVE BUG APPEARS FIXED")
        print("   Engine shows good positional understanding as both colors")
        return 0
    elif len(suspicious_moves) > len(good_moves):
        print("❌ POSSIBLE PERSPECTIVE ISSUE")
        print("   Too many edge/anti-positional moves detected")
        return 1
    else:
        print("⚠️  MIXED RESULTS - NEEDS LARGER SAMPLE")
        print("   Some good moves, some questionable ones")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest error: {e}")
        sys.exit(1)