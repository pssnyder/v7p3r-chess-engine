#!/usr/bin/env python3
"""
Quick tournament test: V14.1 vs V15.2
Tests for the alternating win/loss pattern that indicated perspective bug.

Time control: 8 seconds per move (should reach depth 6-7)
Games: 2 games (one as each color)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import chess.pgn
import time
from datetime import datetime
from v7p3r import V7P3REngine


class EnginePlayer:
    def __init__(self, name: str, engine_class, time_per_move: float = 8.0):
        self.name = name
        self.engine = engine_class()
        self.time_per_move = time_per_move
        
    def get_move(self, board: chess.Board) -> chess.Move:
        """Get move from engine with time limit"""
        start_time = time.time()
        move = self.engine.get_best_move(time_left=self.time_per_move, increment=0.1)
        elapsed = time.time() - start_time
        
        print(f"  {self.name} played {move} in {elapsed:.2f}s")
        return move


def play_game(white_player: EnginePlayer, black_player: EnginePlayer, game_num: int) -> str:
    """Play a single game between two engines"""
    print(f"\n{'='*60}")
    print(f"GAME {game_num}: {white_player.name} (White) vs {black_player.name} (Black)")
    print(f"{'='*60}")
    
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = white_player.name
    game.headers["Black"] = black_player.name
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Event"] = "V14.1 vs V15.2 Perspective Test"
    
    node = game
    move_count = 0
    max_moves = 100  # Limit game length
    
    print(f"Starting position:")
    print(board)
    print()
    
    while not board.is_game_over() and move_count < max_moves:
        move_count += 1
        current_player = white_player if board.turn == chess.WHITE else black_player
        
        print(f"Move {(move_count + 1) // 2}: {current_player.name} to move")
        
        try:
            move = current_player.get_move(board)
            
            if move not in board.legal_moves:
                print(f"  ILLEGAL MOVE: {move}")
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                break
                
            board.push(move)
            node = node.add_variation(move)
            
            # Show position after significant moves
            if move_count <= 10 or move_count % 10 == 0:
                print(f"  Position after {move}:")
                print(f"  {board.fen()}")
                print()
                
        except Exception as e:
            print(f"  ERROR getting move from {current_player.name}: {e}")
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            break
    
    # Determine result
    if board.is_checkmate():
        result = "0-1" if board.turn == chess.WHITE else "1-0"
        print(f"CHECKMATE! Winner: {black_player.name if board.turn == chess.WHITE else white_player.name}")
    elif board.is_stalemate():
        result = "1/2-1/2"
        print("STALEMATE!")
    elif board.is_insufficient_material():
        result = "1/2-1/2"
        print("INSUFFICIENT MATERIAL!")
    elif move_count >= max_moves:
        result = "1/2-1/2"
        print(f"DRAW by move limit ({max_moves} moves)")
    else:
        # Game ended by exception
        print(f"GAME ENDED: {result}")
    
    game.headers["Result"] = result
    
    print(f"\nFinal position ({move_count} moves):")
    print(board)
    print(f"Result: {result}")
    
    return result


def run_tournament():
    """Run a 2-game match between V14.1 and V15.2"""
    print("V7P3R Tournament: V14.1 vs V15.2 Perspective Bug Test")
    print("Time Control: 8 seconds per move")
    print("Expected: No alternating pattern if perspective bug is fixed")
    print()
    
    # Create players
    # Note: Both use same engine class but represent different versions
    # In real tournament, these would be different engine binaries
    v14_1 = EnginePlayer("V7P3R_v14.1", V7P3REngine, 8.0)
    v15_2 = EnginePlayer("V7P3R_v15.2", V7P3REngine, 8.0)
    
    print("WARNING: Both engines using V15.2 code for this test")
    print("(In real tournament, V14.1 would be separate binary)")
    print("This tests V15.2's consistency as both colors")
    print()
    
    results = []
    
    # Game 1: V15.2 as White vs V14.1 as Black
    print("Setting up Game 1...")
    result1 = play_game(v15_2, v14_1, 1)
    results.append(("V15.2-White vs V14.1-Black", result1))
    
    # Brief pause between games
    time.sleep(2)
    
    # Game 2: V14.1 as White vs V15.2 as Black  
    print("Setting up Game 2...")
    result2 = play_game(v14_1, v15_2, 2)
    results.append(("V14.1-White vs V15.2-Black", result2))
    
    # Analyze results
    print(f"\n{'='*60}")
    print("TOURNAMENT RESULTS")
    print(f"{'='*60}")
    
    for game_desc, result in results:
        print(f"{game_desc}: {result}")
    
    # Check for alternating pattern
    print(f"\nPerspective Bug Analysis:")
    print(f"V15.2 as White: {result1}")
    print(f"V15.2 as Black: {result2}")
    
    # Convert results to V15.2's perspective
    v15_2_white_result = result1
    if result2 == "1-0":  # V14.1 won, so V15.2 lost
        v15_2_black_result = "0-1"
    elif result2 == "0-1":  # V14.1 lost, so V15.2 won
        v15_2_black_result = "1-0"
    else:  # Draw
        v15_2_black_result = "1/2-1/2"
    
    print(f"\nV15.2 Performance:")
    print(f"  As White: {v15_2_white_result}")
    print(f"  As Black: {v15_2_black_result}")
    
    # Check for perspective consistency
    if v15_2_white_result == v15_2_black_result:
        if v15_2_white_result == "1/2-1/2":
            print("✓ EXCELLENT: V15.2 drew both games (consistent performance)")
        else:
            print("✓ GOOD: V15.2 had same result as both colors (consistent)")
    else:
        # Different results - check if it's reasonable variation
        white_win = v15_2_white_result == "1-0"
        black_win = v15_2_black_result == "1-0"
        white_loss = v15_2_white_result == "0-1"
        black_loss = v15_2_black_result == "0-1"
        
        if (white_win and black_loss) or (white_loss and black_win):
            print("⚠️  WARNING: Possible perspective issue - opposite results by color")
            print("   (Though this could be normal variation in a 2-game sample)")
        else:
            print("✓ ACCEPTABLE: Different results but no clear color bias")
    
    # Summary
    print(f"\nConclusion:")
    if len(set([v15_2_white_result, v15_2_black_result])) == 1:
        print("✅ No perspective bug detected - consistent performance")
    else:
        print("📊 Results varied by color - monitor in larger sample")
        print("   (2 games is small sample size for definitive conclusion)")
    
    return results


if __name__ == "__main__":
    try:
        results = run_tournament()
        print(f"\nTournament completed successfully!")
    except KeyboardInterrupt:
        print(f"\nTournament interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTournament error: {e}")
        sys.exit(1)