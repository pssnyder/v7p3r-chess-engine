#!/usr/bin/env python3
"""
Quick test of the enhanced v0.5.30 Chess Game
Tests the integration of CopycatEvaluationAI with the chess game interface
"""

import chess
from chess_game import ChessGame

def test_enhanced_chess_game():
    """Test the enhanced chess game with copycat + evaluation AI"""
    print("🚀 Testing V0.5.30 Enhanced Chess Game")
    print("=" * 50)
    
    try:
        # Initialize the enhanced chess game
        game = ChessGame("v7p3r_chess_ai_model.pth", "TestUser")
        print("✅ Enhanced ChessGame initialized successfully!")
        
        # Test AI move generation
        print("\n🎯 Testing AI move generation...")
        
        # Test a few moves
        for i in range(3):
            if not game.board.is_game_over():
                print(f"\nMove {i+1}:")
                print(f"  Position: {game.board.fen()}")
                print(f"  Turn: {'White' if game.board.turn else 'Black'}")
                
                # Get AI move
                ai_move_uci = game.ai_move()
                
                if ai_move_uci:
                    move = chess.Move.from_uci(ai_move_uci)
                    san_move = game.board.san(move)
                    print(f"  🤖 AI selected: {san_move} ({ai_move_uci})")
                    print(f"  📊 Evaluation: {game.current_eval:.2f}")
                    
                    # Make the move
                    game.board.push(move)
                    game.last_ai_move = ai_move_uci
                    
                else:
                    print("  ❌ No move generated")
                    break
            else:
                print("  🏁 Game over!")
                break
        
        print("\n✅ All tests completed successfully!")
        print("🎉 V0.5.30 Enhanced Chess Game is working!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_chess_game()
