import chess
import torch
import numpy as np
import random
from chess_core import ChessAI
import pickle

def load_model(model_path, num_classes):
    """Load a trained model from file"""
    model = ChessAI(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

def board_to_tensor(board):
    """Convert board to tensor representation (12 channels for 6 pieces x 2 colors)"""
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    # Fill tensor based on piece positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Get tensor channel (0-5 for white pieces, 6-11 for black)
            channel = piece.piece_type - 1
            if piece.color == chess.BLACK:
                channel += 6
                
            # Fill the corresponding position in the tensor
            row = 7 - square // 8  # Flip row for correct orientation
            col = square % 8
            tensor[channel][row][col] = 1
    
    return tensor

def play_self_game(model, move_to_index, index_to_move, max_moves=100):
    """Play a complete game of self-play and return the game record"""
    board = chess.Board()
    moves = []
    
    for _ in range(max_moves):
        if board.is_game_over():
            break
        
        # Convert board to tensor
        tensor = board_to_tensor(board)
        tensor = torch.FloatTensor(tensor).unsqueeze(0)  # Add batch dimension
        
        # Use the model to predict move
        with torch.no_grad():
            policy, value = model(tensor)
            
        # Get move probabilities
        probabilities = torch.softmax(policy, dim=1).squeeze(0).numpy()
        
        # Filter to only legal moves
        legal_move_indices = [move_to_index.get(move.uci(), -1) for move in board.legal_moves]
        legal_move_indices = [idx for idx in legal_move_indices if idx >= 0]
        
        if not legal_move_indices:
            # No legal moves found in vocabulary, use fallback
            move = model.select_fallback(board)
        else:
            # Get probabilities of legal moves
            legal_probs = [(idx, probabilities[idx]) for idx in legal_move_indices]
            
            # Sort by probability (descending)
            legal_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Get top moves
            top_moves = legal_probs[:min(3, len(legal_probs))]
            
            # Sample from top moves (weighted by probability)
            weights = [prob for _, prob in top_moves]
            total = sum(weights)
            weights = [w/total for w in weights]
            
            selected_idx = random.choices([idx for idx, _ in top_moves], weights=weights)[0]
            move_uci = index_to_move[selected_idx]
            move = chess.Move.from_uci(move_uci)
        
        # Make the move
        board.push(move)
        moves.append(move)
    
    # Return the game result and moves
    result = board.result()
    return result, moves

def generate_self_play_games(model_path, num_games=100):
    """Generate a set of self-play games for further training"""
    # Load move vocabulary
    with open("move_vocab.pkl", "rb") as f:
        move_to_index = pickle.load(f)
    
    # Create reverse mapping
    index_to_move = {idx: move for move, idx in move_to_index.items()}
    
    # Load model
    model = load_model(model_path, len(move_to_index))
    
    # Generate games
    games = []
    for i in range(num_games):
        result, moves = play_self_game(model, move_to_index, index_to_move)
        games.append((result, moves))
        print(f"Game {i+1}/{num_games}: {result} ({len(moves)} moves)")
    
    return games

if __name__ == "__main__":
    # Example usage
    games = generate_self_play_games("v7p3r_chess_genetic_model.pth", num_games=10)
    
    # Print game statistics
    white_wins = sum(1 for result, _ in games if result == "1-0")
    black_wins = sum(1 for result, _ in games if result == "0-1")
    draws = sum(1 for result, _ in games if result == "1/2-1/2")
    
    print(f"Self-play results: White wins: {white_wins}, Black wins: {black_wins}, Draws: {draws}")
