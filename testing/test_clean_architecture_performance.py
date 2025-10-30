#!/usr/bin/env python3
"""
Performance test for V14.4 clean architecture
Tests unified bitboard evaluator with integrated safety
"""
import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r_bitboard_evaluator import V7P3RBitboardEvaluator
import chess

def test_clean_architecture_performance():
    """Test performance of our cleaned unified architecture"""
    print("Testing Clean Architecture Performance (V14.4)")
    print("=" * 60)
    
    # Standard piece values
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    evaluator = V7P3RBitboardEvaluator(piece_values)
    
    # Test positions
    positions = [
        chess.Board(),  # Starting position
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),  # After 1.e4
        chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3"),  # After 1.e4 e5 2.Nf3
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4"),  # After 1.e4 e5 2.Nf3 Nc6
    ]
    
    print("Warming up evaluator...")
    # Warm up
    for _ in range(10):
        for board in positions:
            evaluator.evaluate_position_complete(board)
    
    print("Running main performance test...")
    # Performance test
    iterations = 1000
    start_time = time.time()
    
    for _ in range(iterations):
        for board in positions:
            score = evaluator.evaluate_position_complete(board)
    
    end_time = time.time()
    total_time = end_time - start_time
    total_evaluations = iterations * len(positions)
    evals_per_second = total_evaluations / total_time
    
    print(f"Architecture Performance Results:")
    print(f"   • Total Evaluations: {total_evaluations:,}")
    print(f"   • Total Time: {total_time:.3f}s")
    print(f"   • Evaluations/Second: {evals_per_second:,.0f}")
    print()
    
    # Test safety integration performance
    print("Testing Integrated Safety Performance")
    print("-" * 40)
    
    start_time = time.time()
    safety_tests = 500
    
    for _ in range(safety_tests):
        for board in positions:
            # Test safety analysis
            safety_analysis = evaluator.analyze_safety_bitboard(board)
            
            # Test move safety evaluation
            legal_moves = list(board.legal_moves)
            if legal_moves:
                move = legal_moves[0]
                move_safety = evaluator.evaluate_move_safety_bitboard(board, move)
    
    end_time = time.time()
    safety_time = end_time - start_time
    safety_evals = safety_tests * len(positions) * 2  # 1 safety + 1 move safety per position
    safety_per_second = safety_evals / safety_time
    
    print(f"   • Safety Evaluations: {safety_evals:,}")
    print(f"   • Safety Time: {safety_time:.3f}s") 
    print(f"   • Safety Evals/Second: {safety_per_second:,.0f}")
    print()
    
    # Test individual components
    print("Component Breakdown Test")
    print("-" * 30)
    
    board = chess.Board()
    component_times = {}
    
    # Test complete evaluation
    start = time.time()
    for _ in range(500):
        complete_eval = evaluator.evaluate_position_complete(board)
    component_times['complete_eval'] = time.time() - start
    
    # Test safety analysis
    start = time.time()
    for _ in range(500):
        safety = evaluator.analyze_safety_bitboard(board)
    component_times['safety_analysis'] = time.time() - start
    
    # Test move safety
    start = time.time()
    legal_moves = list(board.legal_moves)
    if legal_moves:
        test_move = legal_moves[0]
        for _ in range(500):
            move_safety = evaluator.evaluate_move_safety_bitboard(board, test_move)
        component_times['move_safety'] = time.time() - start
    
    for component, duration in component_times.items():
        rate = 500 / duration
        print(f"   • {component}: {rate:,.0f} evals/sec")
    
    print()
    print("Clean Architecture Performance Test Complete!")
    print(f"Overall Performance: {evals_per_second:,.0f} evals/sec")
    print(f"Safety Integration: {safety_per_second:,.0f} safety evals/sec")
    
    # Performance assessment
    if evals_per_second > 3000:
        print("EXCELLENT: Architecture performance maintained!")
    elif evals_per_second > 2500:
        print("GOOD: Architecture performance acceptable")
    else:
        print("ATTENTION: Performance may need optimization")
    
    return evals_per_second, safety_per_second

if __name__ == "__main__":
    test_clean_architecture_performance()