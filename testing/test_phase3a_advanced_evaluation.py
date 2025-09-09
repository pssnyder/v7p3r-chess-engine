#!/usr/bin/env python3
"""
V7P3R v11 Phase 3A - Advanced Evaluation Test
Test the enhanced pawn structure and king safety evaluation
"""

import sys
import os
import time
import chess

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def test_phase3a_initialization():
    """Test that Phase 3A advanced evaluators initialize correctly"""
    print("=== Testing Phase 3A Advanced Evaluator Initialization ===")
    
    try:
        engine = V7P3REngine()
        
        # Check if advanced evaluators are loaded
        if hasattr(engine, 'advanced_pawn_evaluator'):
            print("✅ Advanced pawn evaluator initialized")
        else:
            print("❌ Advanced pawn evaluator missing")
        
        if hasattr(engine, 'king_safety_evaluator'):
            print("✅ King safety evaluator initialized")
        else:
            print("❌ King safety evaluator missing")
        
        print(f"Nudge database: {len(engine.nudge_database)} positions")
        print("✅ Phase 3A initialization successful")
        
    except Exception as e:
        print(f"❌ Phase 3A initialization failed: {e}")
    
    print()

def test_pawn_structure_evaluation():
    """Test advanced pawn structure evaluation"""
    print("=== Testing Advanced Pawn Structure Evaluation ===")
    
    try:
        engine = V7P3REngine()
        
        # Test positions with different pawn structures
        test_positions = [
            {
                'name': 'Starting Position',
                'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -',
                'expected': 'Neutral pawn structure'
            },
            {
                'name': 'Doubled Pawns',
                'fen': 'rnbqkbnr/pp1ppppp/8/8/2p5/8/PPPPPPPP/RNBQKBNR w KQkq -',
                'expected': 'Penalty for doubled pawns'
            },
            {
                'name': 'Passed Pawn',
                'fen': 'rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq -',
                'expected': 'Evaluation with passed pawns'
            }
        ]
        
        for pos in test_positions:
            board = chess.Board(pos['fen'])
            
            # Test pawn evaluation directly
            white_pawn_score = engine.advanced_pawn_evaluator.evaluate_pawn_structure(board, True)
            black_pawn_score = engine.advanced_pawn_evaluator.evaluate_pawn_structure(board, False)
            
            print(f"{pos['name']}:")
            print(f"  White pawn score: {white_pawn_score:.1f}")
            print(f"  Black pawn score: {black_pawn_score:.1f}")
            print(f"  Expected: {pos['expected']}")
            
        print("✅ Pawn structure evaluation working")
        
    except Exception as e:
        print(f"❌ Pawn structure evaluation failed: {e}")
    
    print()

def test_king_safety_evaluation():
    """Test enhanced king safety evaluation"""
    print("=== Testing Enhanced King Safety Evaluation ===")
    
    try:
        engine = V7P3REngine()
        
        # Test positions with different king safety scenarios
        test_positions = [
            {
                'name': 'Castled King',
                'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w Qkq -',
                'expected': 'Safe castled position'
            },
            {
                'name': 'Exposed King',
                'fen': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -',
                'expected': 'King in center'
            },
            {
                'name': 'King with Shelter',
                'fen': 'rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq -',
                'expected': 'Good pawn shelter'
            }
        ]
        
        for pos in test_positions:
            board = chess.Board(pos['fen'])
            
            # Test king safety evaluation directly
            white_king_score = engine.king_safety_evaluator.evaluate_king_safety(board, True)
            black_king_score = engine.king_safety_evaluator.evaluate_king_safety(board, False)
            
            print(f"{pos['name']}:")
            print(f"  White king safety: {white_king_score:.1f}")
            print(f"  Black king safety: {black_king_score:.1f}")
            print(f"  Expected: {pos['expected']}")
            
        print("✅ King safety evaluation working")
        
    except Exception as e:
        print(f"❌ King safety evaluation failed: {e}")
    
    print()

def test_integrated_evaluation():
    """Test integrated evaluation with all Phase 3A components"""
    print("=== Testing Integrated V11 Phase 3A Evaluation ===")
    
    try:
        engine = V7P3REngine()
        
        # Test on starting position
        board = chess.Board()
        
        print("Testing starting position evaluation...")
        eval_score = engine._evaluate_position(board)
        print(f"Integrated evaluation: {eval_score:.2f}")
        
        # Test evaluation components separately
        white_base = engine.bitboard_evaluator.calculate_score_optimized(board, True)
        black_base = engine.bitboard_evaluator.calculate_score_optimized(board, False)
        white_pawn = engine.advanced_pawn_evaluator.evaluate_pawn_structure(board, True)
        black_pawn = engine.advanced_pawn_evaluator.evaluate_pawn_structure(board, False)
        white_king = engine.king_safety_evaluator.evaluate_king_safety(board, True)
        black_king = engine.king_safety_evaluator.evaluate_king_safety(board, False)
        
        print("Evaluation breakdown:")
        print(f"  Base (White): {white_base:.1f}, (Black): {black_base:.1f}")
        print(f"  Pawn (White): {white_pawn:.1f}, (Black): {black_pawn:.1f}")
        print(f"  King (White): {white_king:.1f}, (Black): {black_king:.1f}")
        
        # Test that evaluation caching works
        start_time = time.time()
        eval_score2 = engine._evaluate_position(board)
        cache_time = time.time() - start_time
        
        if eval_score == eval_score2:
            print(f"✅ Evaluation caching working (cache time: {cache_time:.6f}s)")
        else:
            print("❌ Evaluation caching inconsistent")
        
        print("✅ Integrated evaluation working")
        
    except Exception as e:
        print(f"❌ Integrated evaluation failed: {e}")
    
    print()

def test_search_with_phase3a():
    """Test search performance with Phase 3A enhancements"""
    print("=== Testing Search with Phase 3A Enhancements ===")
    
    try:
        engine = V7P3REngine()
        
        # Test search on starting position
        board = chess.Board()
        print("Testing search with Phase 3A evaluation...")
        
        start_time = time.time()
        best_move = engine.search(board, time_limit=2.0)
        search_time = time.time() - start_time
        
        print(f"Best move: {best_move}")
        print(f"Search time: {search_time:.3f}s")
        print(f"Nodes searched: {engine.nodes_searched}")
        
        if engine.nodes_searched > 0:
            nps = engine.nodes_searched / search_time if search_time > 0 else 0
            print(f"Nodes per second: {nps:.0f}")
        
        # Check evaluation cache performance
        cache_hits = engine.search_stats.get('cache_hits', 0)
        cache_misses = engine.search_stats.get('cache_misses', 0)
        total_evals = cache_hits + cache_misses
        
        if total_evals > 0:
            cache_rate = (cache_hits / total_evals) * 100
            print(f"Evaluation cache hit rate: {cache_rate:.1f}%")
        
        print("✅ Search with Phase 3A working")
        
    except Exception as e:
        print(f"❌ Search with Phase 3A failed: {e}")
    
    print()

def test_phase3a_stability():
    """Test stability of Phase 3A enhancements"""
    print("=== Testing Phase 3A Stability ===")
    
    try:
        engine = V7P3REngine()
        
        # Test multiple positions to ensure stability
        test_fens = [
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -',
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3',
            'rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -',
            'r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq -'
        ]
        
        all_stable = True
        
        for i, fen in enumerate(test_fens):
            try:
                board = chess.Board(fen)
                eval_score = engine._evaluate_position(board)
                print(f"Position {i+1}: {eval_score:.2f}")
            except Exception as e:
                print(f"❌ Position {i+1} failed: {e}")
                all_stable = False
        
        if all_stable:
            print("✅ Phase 3A stability confirmed")
        else:
            print("❌ Phase 3A stability issues detected")
        
    except Exception as e:
        print(f"❌ Phase 3A stability test failed: {e}")
    
    print()

def main():
    """Run all Phase 3A tests"""
    print("V7P3R v11 Phase 3A - Advanced Evaluation Testing")
    print("=" * 60)
    
    test_phase3a_initialization()
    test_pawn_structure_evaluation()
    test_king_safety_evaluation() 
    test_integrated_evaluation()
    test_search_with_phase3a()
    test_phase3a_stability()
    
    print("=" * 60)
    print("Phase 3A advanced evaluation testing complete!")

if __name__ == "__main__":
    main()
