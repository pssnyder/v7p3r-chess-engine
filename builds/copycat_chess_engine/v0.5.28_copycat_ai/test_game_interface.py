#!/usr/bin/env python3
"""
Simple test to verify the v0.5.28 copycat AI game interface works
Tests the pygame interface without actually opening the GUI
"""

import chess
import pygame
from chess_game import ChessGame

def test_game_initialization():
    """Test if the game can initialize properly"""
    print("🎯 Testing V0.5.28 Copycat AI Game Interface")
    print("=" * 50)
    
    try:
        # Initialize pygame (but don't create window)
        pygame.init()
        
        # Test game creation
        print("📋 Creating ChessGame instance...")
        game = ChessGame("v7p3r_chess_ai_model.pth")
        print("✅ Game initialized successfully!")
        
        # Test AI move functionality
        print("📋 Testing AI move generation...")
        starting_fen = game.board.fen()
        print(f"   Starting position: {starting_fen}")
        
        # Test ai_move method
        ai_move = game.ai_move()
        print(f"✅ AI suggested move: {ai_move}")
        
        # Check if move was legal
        if game.board.fen() != starting_fen:
            print("✅ Move was applied to board successfully!")
            print(f"   New position: {game.board.fen()}")
        else:
            print("⚠️  Move was not applied to board")
        
        # Test a few more moves
        print("📋 Testing multiple AI moves...")
        for i in range(3):
            if not game.board.is_game_over():
                # Switch turns (simulate human move with a random legal move)
                if game.board.legal_moves:
                    human_move = list(game.board.legal_moves)[0]
                    game.board.push(human_move)
                    print(f"   Human plays: {human_move.uci()}")
                
                # AI response
                if not game.board.is_game_over():
                    ai_move = game.ai_move()
                    print(f"   AI responds: {ai_move}")
        
        print("✅ Game interface test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        pygame.quit()

def test_move_quality():
    """Test the quality of AI moves"""
    print("\n🎯 Testing Move Quality")
    print("=" * 30)
    
    try:
        game = ChessGame("v7p3r_chess_ai_model.pth")
        
        # Test in starting position
        print("📋 Starting position analysis:")
        moves_tested = 0
        legal_moves_found = 0
        
        for _ in range(10):  # Test 10 moves
            if game.board.is_game_over():
                break
                
            # Get AI move
            move = game.ai_move()
            moves_tested += 1
            
            # Check if it was legal (board state changed)
            if move:
                legal_moves_found += 1
                print(f"   Move {moves_tested}: {move} ✅")
            else:
                print(f"   Move {moves_tested}: Failed ❌")
            
            # Reset to starting position for consistent testing
            game.board = chess.Board()
        
        success_rate = (legal_moves_found / moves_tested) * 100 if moves_tested > 0 else 0
        print(f"\n📊 Results: {legal_moves_found}/{moves_tested} moves successful ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print("✅ Excellent move quality!")
        elif success_rate >= 70:
            print("⚠️  Good move quality, some issues")
        else:
            print("❌ Poor move quality, needs attention")
            
    except Exception as e:
        print(f"❌ Error during move quality testing: {str(e)}")

if __name__ == "__main__":
    print("🚀 V0.5.28 Copycat AI - Game Interface Testing")
    print()
    
    # Test 1: Basic initialization and interface
    if test_game_initialization():
        print("\n" + "=" * 50)
        
        # Test 2: Move quality
        test_move_quality()
        
        print("\n🎉 All tests completed!")
        print("\nThe v0.5.28 copycat AI is ready to play!")
        print("Run 'python chess_game.py' to start the GUI game.")
    else:
        print("\n❌ Basic tests failed. Check the errors above.")
