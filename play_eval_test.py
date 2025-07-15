"""
Play a console-based chess game between V7P3R and Stockfish.
This script provides detailed evaluation information after each move
to help analyze the engine's decision-making process.
"""

import chess
import chess.pgn
import time
import os
import datetime
import json
import sys

def load_config():
    """Load configuration from config.json"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return {
            "stockfish_path": "stockfish.exe",
            "stockfish_elo": 400,
            "search_depth": 3,
            "move_time": 1.0
        }

def count_material(board):
    """Count material for both sides and return the difference"""
    piece_values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.25,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0
    }
    
    white_material = sum(len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type] 
                         for piece_type in piece_values)
    black_material = sum(len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
                         for piece_type in piece_values)
    
    return white_material - black_material

def main():
    config = load_config()
    
    # Import required modules
    from v7p3r_engine import V7P3REngine
    from v7p3r_stockfish import StockfishHandler
    from v7p3r_config import V7P3RConfig
    
    # Initialize config
    v7p3r_config = V7P3RConfig('config.json')
    
    # Initialize engines
    v7p3r = V7P3REngine('config.json')
    stockfish = StockfishHandler(v7p3r_config)
    
    # Initialize board
    board = chess.Board()
    
    # Create PGN game record
    game = chess.pgn.Game()
    game.headers["Event"] = "V7P3R Evaluation Test"
    game.headers["Site"] = "Console"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "V7P3R"
    game.headers["Black"] = "Stockfish"
    
    # Play game
    node = game
    move_count = 0
    
    print("\n=== V7P3R Chess Engine Evaluation Test ===")
    print("White: V7P3R | Black: Stockfish")
    print("Starting new game...\n")
    
    try:
        while not board.is_game_over():
            # Print current board
            print(board)
            
            # Calculate material difference
            material_diff = count_material(board)
            
            # V7P3R's turn (White)
            if board.turn == chess.WHITE:
                print("\nV7P3R's turn (White)...")
                
                # Get detailed evaluations for all moves
                start_time = time.time()
                best_move = v7p3r.find_move(board)
                search_time = time.time() - start_time
                
                if best_move is None:
                    print("  Error: V7P3R could not find a legal move")
                    break
                
                # Get evaluation for current position
                evaluation = v7p3r.get_evaluation(board)
                
                # Print move info
                print(f"  Move: {board.san(best_move)}")
                print(f"  V7P3R Evaluation: {evaluation:.2f}")
                print(f"  Material Difference: {material_diff:.2f}")
                print(f"  Search Time: {search_time:.2f}s")
                
                # Make the move
                move = best_move
            
            # Stockfish's turn (Black)
            else:
                print("\nStockfish's turn (Black)...")
                
                # Get move from Stockfish
                start_time = time.time()
                move = stockfish.get_move(board)
                search_time = time.time() - start_time
                
                if move is None:
                    print("  Error: Stockfish could not find a legal move")
                    break
                
                # Print move info
                print(f"  Move: {board.san(move)}")
                print(f"  Material Difference: {material_diff:.2f}")
                print(f"  Search Time: {search_time:.2f}s")
            
            # Make the move
            if move:
                node = node.add_variation(move)
                board.push(move)
                move_count += 1
            else:
                print("Invalid move, ending game")
                break
            
            # Print separator
            print("-" * 40)
        
        # Game over
        print("\n=== Game Over ===")
        print(board)
        print(f"Result: {board.result()}")
        
        # Save game to PGN
        game.headers["Result"] = board.result()
        
        os.makedirs("pgn_game_records", exist_ok=True)
        pgn_file = f"pgn_game_records/eval_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
        
        with open(pgn_file, "w") as f:
            print(game, file=f, end="\n\n")
        
        print(f"Game saved to {pgn_file}")
    
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    except Exception as e:
        print(f"\nError during game: {str(e)}")
    finally:
        # Clean up (Stockfish cleanup)
        try:
            if hasattr(stockfish, 'quit'):
                stockfish.quit()
        except:
            pass
        
        print("\nGame session ended.")

if __name__ == "__main__":
    main()
