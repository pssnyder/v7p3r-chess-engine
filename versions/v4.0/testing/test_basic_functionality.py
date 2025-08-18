# testing/test_basic_functionality.py

"""Basic Functionality Tests for V7P3R Chess Engine
Tests core engine functionality and components.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import time
from v7p3r_engine import V7P3REngine
from v7p3r_config import V7P3RConfig
from v7p3r_stockfish import StockfishHandler

def test_engine_initialization():
    """Test that the engine initializes correctly"""
    print("Testing engine initialization...")
    
    try:
        config = V7P3RConfig("config.json")
        engine = V7P3REngine("config.json")
        
        info = engine.get_engine_info()
        print(f"✓ Engine initialized: {info['name']} v{info['version']}")
        print(f"  - Search algorithm: {info['search_algorithm']}")
        print(f"  - Max depth: {info['max_depth']}")
        print(f"  - Book positions: {info['book_positions']}")
        
        return True
    except Exception as e:
        print(f"✗ Engine initialization failed: {e}")
        return False

def test_move_generation():
    """Test move generation and evaluation"""
    print("\nTesting move generation...")
    
    try:
        engine = V7P3REngine("config.json")
        board = chess.Board()
        
        # Test initial position
        move = engine.find_move(board, time_limit=5.0)
        if move and move in board.legal_moves:
            print(f"✓ Generated legal move: {move}")
            
            # Test move analysis
            analysis = engine.get_position_analysis(board)
            print(f"  - Position evaluation: {analysis['evaluation']}")
            print(f"  - In opening book: {analysis['in_book']}")
            print(f"  - Game phase: {analysis['guidelines']['phase']}")
            
            return True
        else:
            print(f"✗ Invalid move generated: {move}")
            return False
            
    except Exception as e:
        print(f"✗ Move generation failed: {e}")
        return False

def test_stockfish_integration():
    """Test Stockfish integration"""
    print("\nTesting Stockfish integration...")
    
    try:
        config = V7P3RConfig("config.json")
        stockfish = StockfishHandler(config)
        
        if stockfish.is_available():
            board = chess.Board()
            move = stockfish.get_move(board)
            
            if move and move in board.legal_moves:
                print(f"✓ Stockfish generated legal move: {move}")
                
                # Test evaluation
                evaluation = stockfish.get_evaluation(board)
                print(f"  - Stockfish evaluation: {evaluation}")
                
                return True
            else:
                print(f"✗ Stockfish generated invalid move: {move}")
                return False
        else:
            print("✗ Stockfish not available")
            return False
            
    except Exception as e:
        print(f"✗ Stockfish integration failed: {e}")
        return False

def test_performance():
    """Test engine performance"""
    print("\nTesting engine performance...")
    
    try:
        engine = V7P3REngine("config.json")
        board = chess.Board()
        
        # Test multiple positions for average performance
        total_time = 0
        moves_tested = 0
        max_time = 0
        
        positions = [
            chess.Board(),  # Starting position
            chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),  # After 1.e4 e5 2.Nf3 Nc6
            chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),  # After 1.e4 d5
        ]
        
        for i, test_board in enumerate(positions):
            start_time = time.time()
            move = engine.find_move(test_board, time_limit=10.0)
            move_time = time.time() - start_time
            
            if move:
                total_time += move_time
                moves_tested += 1
                max_time = max(max_time, move_time)
                print(f"  Position {i+1}: {move} ({move_time:.2f}s)")
        
        if moves_tested > 0:
            avg_time = total_time / moves_tested
            print(f"✓ Performance test completed")
            print(f"  - Average time: {avg_time:.2f}s")
            print(f"  - Maximum time: {max_time:.2f}s")
            print(f"  - Moves tested: {moves_tested}")
            
            # Check if performance meets requirements
            if avg_time < 30.0 and max_time < 30.0:
                print("✓ Performance meets requirements (< 30s per move)")
                return True
            else:
                print("⚠ Performance may not meet requirements")
                return False
        else:
            print("✗ No moves generated for performance test")
            return False
            
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False

def test_game_completion():
    """Test that a short game can complete"""
    print("\nTesting game completion...")
    
    try:
        engine = V7P3REngine("config.json")
        config = V7P3RConfig("config.json")
        stockfish = StockfishHandler(config)
        
        if not stockfish.is_available():
            print("⚠ Stockfish not available for game test")
            return True
        
        board = chess.Board()
        moves_played = 0
        max_moves = 10  # Short test game
        
        while not board.is_game_over() and moves_played < max_moves:
            if board.turn == chess.WHITE:
                # V7P3R plays white
                move = engine.find_move(board, time_limit=5.0)
                player = "V7P3R"
            else:
                # Stockfish plays black
                move = stockfish.get_move(board)
                player = "Stockfish"
            
            if move and move in board.legal_moves:
                board.push(move)
                moves_played += 1
                print(f"  Move {moves_played}: {player} played {move}")
            else:
                print(f"✗ Invalid move from {player}: {move}")
                return False
        
        print(f"✓ Game test completed ({moves_played} moves)")
        return True
        
    except Exception as e:
        print(f"✗ Game completion test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== V7P3R Chess Engine Basic Tests ===\n")
    
    tests = [
        test_engine_initialization,
        test_move_generation,
        test_stockfish_integration,
        test_performance,
        test_game_completion
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Engine is ready.")
        return 0
    else:
        print("⚠ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
