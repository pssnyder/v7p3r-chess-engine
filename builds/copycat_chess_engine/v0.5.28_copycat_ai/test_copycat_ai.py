#!/usr/bin/env python3
"""
Test script for V0.5.28 Copycat AI
Tests the trained model's ability to suggest moves that mimic v7p3r's playing style
"""

import chess
import torch
import pickle
import numpy as np
from chess_core import ChessAI

def load_copycat_ai():
    """Load the trained copycat AI model and vocabulary"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load move vocabulary
    with open("move_vocab.pkl", "rb") as f:
        move_to_index = pickle.load(f)
    
    # Create reverse mapping
    index_to_move = {idx: move for move, idx in move_to_index.items()}
    
    # Load model
    model = ChessAI(len(move_to_index)).to(device)
    model.load_state_dict(torch.load("v7p3r_chess_ai_model.pth", map_location=device, weights_only=False))
    model.eval()
    
    return model, move_to_index, index_to_move, device

def board_to_tensor(board):
    """Convert chess board to tensor format (same as training)"""
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
            tensor[channel][7 - square//8][square%8] = 1
    return tensor

def get_copycat_move(model, board, move_to_index, index_to_move, device, top_k=5):
    """Get the copycat AI's suggested moves for a position"""
    # Convert board to tensor
    position_tensor = torch.FloatTensor(board_to_tensor(board)).unsqueeze(0).to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(position_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(index_to_move)))
    
    suggestions = []
    legal_moves = set(move.uci() for move in board.legal_moves)
    
    for i in range(top_k):
        if i < len(top_indices[0]):
            idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            move_uci = index_to_move.get(idx, "unknown")
            
            # Check if move is legal in current position
            is_legal = move_uci in legal_moves
            
            suggestions.append({
                'move': move_uci,
                'probability': prob,
                'legal': is_legal
            })
    
    return suggestions

def test_opening_positions():
    """Test the copycat AI on common opening positions"""
    print("🎯 Testing V0.5.28 Copycat AI - Opening Positions")
    print("=" * 60)
    
    model, move_to_index, index_to_move, device = load_copycat_ai()
    
    test_positions = [
        ("Starting position", chess.Board()),
        ("After 1.e4", chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")),
        ("After 1.e4 e5", chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")),
        ("After 1.d4", chess.Board("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1")),
    ]
    
    for description, board in test_positions:
        print(f"\n📍 {description}")
        print(f"   Turn: {'White' if board.turn else 'Black'}")
        print(f"   FEN: {board.fen()}")
        
        suggestions = get_copycat_move(model, board, move_to_index, index_to_move, device)
        
        print("   🤖 Copycat AI Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            status = "✅ LEGAL" if suggestion['legal'] else "❌ ILLEGAL"
            print(f"      {i}. {suggestion['move']} (prob: {suggestion['probability']:.3f}) {status}")
        
        # Find best legal move
        legal_suggestions = [s for s in suggestions if s['legal']]
        if legal_suggestions:
            best = legal_suggestions[0]
            try:
                move = chess.Move.from_uci(best['move'])
                san_move = board.san(move)
                print(f"   🎯 Best Legal Move: {san_move} ({best['move']}) - {best['probability']:.1%} confidence")
            except:
                print(f"   🎯 Best Legal Move: {best['move']} - {best['probability']:.1%} confidence")
        else:
            print("   ⚠️  No legal moves found in top suggestions!")

def test_interactive_mode():
    """Interactive mode to test positions"""
    print("\n" + "=" * 60)
    print("🎮 Interactive Testing Mode")
    print("Enter FEN positions to see what the copycat AI suggests")
    print("(Press Enter with empty input to skip)")
    print("=" * 60)
    
    model, move_to_index, index_to_move, device = load_copycat_ai()
    
    while True:
        fen_input = input("\nEnter FEN position (or 'quit' to exit): ").strip()
        
        if fen_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not fen_input:
            break
            
        try:
            board = chess.Board(fen_input)
            print(f"\n📍 Position: {board.fen()}")
            print(f"   Turn: {'White' if board.turn else 'Black'}")
            
            suggestions = get_copycat_move(model, board, move_to_index, index_to_move, device)
            
            print("   🤖 Copycat AI Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                status = "✅ LEGAL" if suggestion['legal'] else "❌ ILLEGAL"
                print(f"      {i}. {suggestion['move']} (prob: {suggestion['probability']:.3f}) {status}")
            
            # Find best legal move
            legal_suggestions = [s for s in suggestions if s['legal']]
            if legal_suggestions:
                best = legal_suggestions[0]
                try:
                    move = chess.Move.from_uci(best['move'])
                    san_move = board.san(move)
                    print(f"   🎯 Recommended: {san_move} ({best['move']}) - {best['probability']:.1%} confidence")
                except:
                    print(f"   🎯 Recommended: {best['move']} - {best['probability']:.1%} confidence")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🚀 V0.5.28 Copycat AI Testing Suite")
    print("Testing the AI's ability to mimic v7p3r's playing style")
    
    try:
        # Test standard opening positions
        test_opening_positions()
        
        # Interactive testing
        test_interactive_mode()
        
        print("\n✅ Testing complete!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
