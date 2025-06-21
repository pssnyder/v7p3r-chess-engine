# engine_utilities/best_move_finder.py
import chess
import sys
import os

# Add the parent directory to sys.path for direct script execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from v7p3r import V7P3REvaluationEngine

def get_sorted_legal_moves(board, engine, player):
    """
    Returns all legal moves sorted by evaluation score.
    """
    legal_moves = list(board.legal_moves)
    scored_moves = []
    
    for move in legal_moves:
        # Make the move on a copy of the board
        board_copy = board.copy()
        board_copy.push(move)
        
        # Evaluate the resulting position
        score = engine.evaluate(board_copy, player)
        scored_moves.append((move, score))
    
    # Sort moves by score (best to worst)
    if player == chess.WHITE:
        scored_moves.sort(key=lambda x: x[1], reverse=True)  # Higher is better for white
    else:
        scored_moves.sort(key=lambda x: x[1])  # Lower is better for black
    
    return [move for move, _ in scored_moves]

def find_best_move(FEN, depth=3, ai_type='deepsearch'):
    try:
        board = chess.Board()
        board.set_fen(FEN)
        
        # Get the player to move from the board state
        player = board.turn
        
        # Initialize the engine with the specified settings
        engine = V7P3REvaluationEngine()
        engine.ai_type = ai_type
        engine.depth = depth
        
        # Search for the best move
        best_move = engine.search(board.copy(), player)
        
        # Check if the move is legal
        if best_move is not None and best_move in board.legal_moves:
            return best_move
        else:
            print(f"Warning: Engine suggested illegal move: {best_move}. Finding alternative...")
            
            # Get sorted legal moves as fallback
            sorted_legal_moves = get_sorted_legal_moves(board, engine, player)
            
            if sorted_legal_moves:
                return sorted_legal_moves[0]  # Return the best legal move
            else:
                return None  # No legal moves available
                
    except Exception as e:
        print(f"Error finding best move: {e}")
        return None

# Example usage
if __name__ == "__main__":
    FEN = input("Enter FEN string: ")
    if not FEN:
        FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Default starting position   
    
    depth_input = input("Enter search depth [default: 3]: ")
    depth = int(depth_input) if depth_input.isdigit() else 3
    
    best_move = find_best_move(FEN, depth)
    
    if best_move:
        # Create a board to get player information for display
        board = chess.Board(FEN)
        player_str = "White" if board.turn == chess.WHITE else "Black"
        print(f"Player to move: {player_str}")
        print(f"Best move: {best_move}")
    else:
        print("Failed to find a valid move.")