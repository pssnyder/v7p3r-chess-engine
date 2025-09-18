#!/usr/bin/env python3
"""
Test V7P3R Position Posture Assessment System
Validates volatility detection and posture determination
"""

import sys
import os
import chess
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r_posture_assessment import V7P3RPostureAssessment, PositionVolatility, GamePosture

def test_posture_assessment():
    """Test posture assessment on various position types"""
    
    print("V7P3R Posture Assessment Test")
    print("=" * 50)
    
    assessor = V7P3RPostureAssessment()
    
    # Test positions with expected characteristics
    test_positions = [
        {
            'name': 'Opening Position (Stable)',
            'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'expected_volatility': PositionVolatility.STABLE,
            'expected_posture': GamePosture.BALANCED
        },
        {
            'name': 'Quiet Middlegame (Stable)',
            'fen': 'rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3',
            'expected_volatility': PositionVolatility.STABLE,
            'expected_posture': GamePosture.BALANCED
        },
        {
            'name': 'Tactical Position (Volatile)',
            'fen': 'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5',
            'expected_volatility': PositionVolatility.TACTICAL,
            'expected_posture': None  # Will vary based on analysis
        },
        {
            'name': 'King Under Attack (Critical)',
            'fen': 'rnbq1rk1/ppp2ppp/3p1n2/4p3/1bPP4/2N1PN2/PP3PPP/R1BQKB1R b KQ - 0 7',
            'expected_volatility': PositionVolatility.TACTICAL,
            'expected_posture': GamePosture.DEFENSIVE
        },
        {
            'name': 'Material Advantage (Offensive)',
            'fen': 'rnbqkb1r/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3',
            'expected_volatility': PositionVolatility.STABLE,
            'expected_posture': GamePosture.BALANCED
        }
    ]
    
    total_time = 0
    successful_tests = 0
    
    for i, test_case in enumerate(test_positions):
        print(f"\nTest {i+1}: {test_case['name']}")
        print(f"FEN: {test_case['fen']}")
        
        board = chess.Board(test_case['fen'])
        
        start_time = time.time()
        volatility, posture = assessor.assess_position_posture(board)
        end_time = time.time()
        
        assessment_time = end_time - start_time
        total_time += assessment_time
        
        print(f"Detected Volatility: {volatility.value}")
        print(f"Detected Posture: {posture.value}")
        print(f"Assessment Time: {assessment_time*1000:.2f}ms")
        
        # Debug: Show detailed analysis for mismatches
        if (test_case['expected_volatility'] and volatility != test_case['expected_volatility']) or \
           (test_case['expected_posture'] and posture != test_case['expected_posture']):
            print(f"DEBUG INFO:")
            # Get some internal metrics for debugging
            attacked_pieces = assessor._count_attacked_pieces(board)
            undefended_pieces = assessor._count_undefended_pieces(board)
            our_threats = assessor._count_our_threats(board)
            their_threats = assessor._count_their_threats(board)
            material_advantage = assessor._get_material_advantage(board)
            
            print(f"  Attacked pieces: {attacked_pieces}")
            print(f"  Undefended pieces: {undefended_pieces}")
            print(f"  Our threats: {our_threats}")
            print(f"  Their threats: {their_threats}")
            print(f"  Material advantage: {material_advantage}")
        
        # Validate results
        volatility_correct = (test_case['expected_volatility'] is None or 
                            volatility == test_case['expected_volatility'])
        posture_correct = (test_case['expected_posture'] is None or 
                         posture == test_case['expected_posture'])
        
        if volatility_correct and posture_correct:
            print("✓ Assessment matches expectations")
            successful_tests += 1
        else:
            print("✗ Assessment differs from expectations")
            if not volatility_correct:
                print(f"  Expected volatility: {test_case['expected_volatility']}")
            if not posture_correct:
                print(f"  Expected posture: {test_case['expected_posture']}")
    
    # Performance summary
    avg_time = (total_time / len(test_positions)) * 1000
    print(f"\n" + "=" * 50)
    print("PERFORMANCE SUMMARY:")
    print(f"Total tests: {len(test_positions)}")
    print(f"Successful: {successful_tests}/{len(test_positions)}")
    print(f"Average assessment time: {avg_time:.2f}ms")
    print(f"Total time: {total_time*1000:.2f}ms")
    
    # Cache statistics
    cache_stats = assessor.get_cache_stats()
    print(f"\nCACHE STATISTICS:")
    print(f"Total calls: {cache_stats['calls']}")
    print(f"Cache hits: {cache_stats['cache_hits']}")
    print(f"Hit rate: {cache_stats['hit_rate_percent']:.1f}%")
    print(f"Cache size: {cache_stats['cache_size']}")
    
    # Performance target: under 5ms per assessment
    performance_target = 5.0  # ms
    performance_pass = avg_time <= performance_target
    
    print(f"\nPerformance Target: {performance_target}ms")
    print(f"Performance Result: {'PASS' if performance_pass else 'FAIL'}")
    
    return successful_tests == len(test_positions) and performance_pass

def test_posture_consistency():
    """Test that posture assessment is consistent for same position"""
    
    print(f"\n" + "=" * 50)
    print("CONSISTENCY TEST:")
    print("=" * 50)
    
    assessor = V7P3RPostureAssessment()
    
    # Test same position multiple times
    board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3")
    
    results = []
    for i in range(5):
        volatility, posture = assessor.assess_position_posture(board)
        results.append((volatility, posture))
    
    # Check consistency
    first_result = results[0]
    consistent = all(result == first_result for result in results)
    
    print(f"Performed 5 assessments of same position")
    print(f"All results identical: {consistent}")
    print(f"Result: {first_result[0].value}, {first_result[1].value}")
    
    # Check cache efficiency
    cache_stats = assessor.get_cache_stats()
    expected_hits = 4  # First call misses, next 4 should hit
    cache_efficient = cache_stats['cache_hits'] >= expected_hits
    
    print(f"Cache hits: {cache_stats['cache_hits']}/4 expected")
    print(f"Cache efficiency: {'PASS' if cache_efficient else 'FAIL'}")
    
    return consistent and cache_efficient

if __name__ == "__main__":
    print("Starting V7P3R Posture Assessment Tests...")
    
    try:
        # Run main tests
        basic_test_pass = test_posture_assessment()
        
        # Run consistency test
        consistency_test_pass = test_posture_consistency()
        
        print(f"\n" + "=" * 50)
        print("FINAL TEST RESULTS:")
        print(f"Basic Assessment Test: {'PASS' if basic_test_pass else 'FAIL'}")
        print(f"Consistency Test: {'PASS' if consistency_test_pass else 'FAIL'}")
        
        overall_pass = basic_test_pass and consistency_test_pass
        print(f"Overall Result: {'PASS' if overall_pass else 'FAIL'}")
        
        if overall_pass:
            print("\n✓ Posture assessment system is ready for integration!")
        else:
            print("\n✗ Posture assessment system needs improvements.")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()