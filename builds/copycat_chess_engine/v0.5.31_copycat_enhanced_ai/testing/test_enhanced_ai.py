#!/usr/bin/env python3
"""
Test script for V0.5.30 Copycat + Evaluation AI
Tests the enhanced system that combines neural network move prediction with evaluation-based selection
"""

import chess
import torch
from chess_core import CopycatEvaluationAI

def test_copycat_evaluation_ai():
    """Test the enhanced copycat + evaluation AI system"""
    print("🚀 Testing V0.5.30 Copycat + Evaluation AI")
    print("=" * 60)
    
    # Initialize the enhanced AI
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    try:
        ai = CopycatEvaluationAI(
            model_path="v7p3r_chess_ai_model.pth",
            vocab_path="move_vocab.pkl", 
            device=device
        )
        print("✅ Enhanced AI initialized successfully!")
        
        # Test positions
        test_positions = [
            ("Starting position", chess.Board()),
            ("After 1.e4", chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")),
            ("After 1.e4 e5", chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")),
        ]
        
        for description, board in test_positions:
            print(f"\n📍 {description}")
            print(f"   Turn: {'White' if board.turn else 'Black'}")
            
            # Get move with debug info
            move = ai.select_best_move(board, top_k=5, debug=True)
            
            if move:
                san_move = board.san(move)
                print(f"   🎯 Selected Move: {san_move} ({move.uci()})")
            else:
                print(f"   ❌ No move selected")
                
            # Get detailed analysis
            analysis = ai.get_move_analysis(board, top_k=5)
            
            if "error" not in analysis:
                print(f"   📊 Analysis Summary:")
                for i, candidate in enumerate(analysis["analysis"][:3]):
                    marker = "👑" if i == 0 else f"{i+1}."
                    print(f"      {marker} {candidate['uci']} - Neural: {candidate['neural_confidence']:.3f}, Eval: {candidate['eval_score']:.2f}")
        
        print("\n✅ All tests completed successfully!")
        
        # Interactive test
        print("\n" + "=" * 60)
        print("🎮 Interactive Mode - Enter FEN positions to test")
        print("(Press Enter to skip, 'quit' to exit)")
        print("=" * 60)
        
        while True:
            fen_input = input("\nEnter FEN position: ").strip()
            
            if fen_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not fen_input:
                break
                
            try:
                board = chess.Board(fen_input)
                print(f"\n📍 Position: {board.fen()}")
                print(f"   Turn: {'White' if board.turn else 'Black'}")
                
                move = ai.select_best_move(board, top_k=5, debug=True)
                
                if move:
                    san_move = board.san(move)
                    print(f"   🎯 Recommended: {san_move} ({move.uci()})")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        print("\n🎉 Enhanced copycat AI testing complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_copycat_evaluation_ai()
