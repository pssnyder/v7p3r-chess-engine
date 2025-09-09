#!/usr/bin/env python3
"""
V7P3R v11 Phase 2 Enhancement - Instant Nudge Move Test
Test the instant nudge move threshold system
"""

import sys
import os
import time
import chess

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def test_instant_nudge_configuration():
    """Test nudge threshold configuration"""
    print("=== Testing Instant Nudge Configuration ===")
    engine = V7P3REngine()
    
    print(f"Instant nudge config: {engine.nudge_instant_config}")
    print(f"Min frequency: {engine.nudge_instant_config['min_frequency']}")
    print(f"Min eval: {engine.nudge_instant_config['min_eval']}")
    print(f"Confidence threshold: {engine.nudge_instant_config['confidence_threshold']}")
    
    print("✅ Configuration loaded successfully")
    print()

def test_instant_nudge_detection():
    """Test instant nudge move detection"""
    print("=== Testing Instant Nudge Detection ===")
    engine = V7P3REngine()
    
    if len(engine.nudge_database) == 0:
        print("❌ No nudge database loaded")
        return
    
    # Find positions with high-confidence moves
    high_confidence_positions = []
    
    for position_key, position_data in engine.nudge_database.items():
        try:
            fen = position_data.get('fen')
            if not fen:
                continue
                
            moves_data = position_data.get('moves', {})
            for move_uci, move_data in moves_data.items():
                frequency = move_data.get('frequency', 0)
                evaluation = move_data.get('eval', 0.0)
                
                # Check if this move meets instant criteria
                if (frequency >= engine.nudge_instant_config['min_frequency'] and 
                    evaluation >= engine.nudge_instant_config['min_eval']):
                    
                    confidence = frequency + (evaluation * 10)
                    if confidence >= engine.nudge_instant_config['confidence_threshold']:
                        high_confidence_positions.append({
                            'fen': fen,
                            'move': move_uci,
                            'frequency': frequency,
                            'eval': evaluation,
                            'confidence': confidence
                        })
                        break  # One per position is enough for testing
        except:
            continue
    
    print(f"Found {len(high_confidence_positions)} high-confidence positions")
    
    # Test instant detection on these positions
    for i, pos_data in enumerate(high_confidence_positions[:3]):  # Test first 3
        print(f"\nTesting position {i+1}:")
        print(f"FEN: {pos_data['fen']}")
        print(f"Expected instant move: {pos_data['move']}")
        print(f"Frequency: {pos_data['frequency']}, Eval: {pos_data['eval']:.3f}, Confidence: {pos_data['confidence']:.1f}")
        
        try:
            board = chess.Board(pos_data['fen'])
            instant_move = engine._check_instant_nudge_move(board)
            
            if instant_move:
                print(f"✅ Instant move detected: {instant_move}")
                if instant_move.uci() == pos_data['move']:
                    print("✅ Correct move selected")
                else:
                    print(f"⚠️ Different move selected (expected {pos_data['move']})")
            else:
                print("❌ No instant move detected")
        except Exception as e:
            print(f"❌ Error testing position: {e}")
    
    print()

def test_search_with_instant_nudges():
    """Test search with instant nudge moves"""
    print("=== Testing Search with Instant Nudges ===")
    engine = V7P3REngine()
    
    # Test starting position (likely has high-frequency moves)
    board = chess.Board()
    print("Testing starting position...")
    
    # Check if starting position has instant nudge
    instant_move = engine._check_instant_nudge_move(board)
    if instant_move:
        print(f"Starting position has instant nudge: {instant_move}")
        
        # Test search - should return instantly
        start_time = time.time()
        best_move = engine.search(board, time_limit=3.0)
        search_time = time.time() - start_time
        
        print(f"Search result: {best_move}")
        print(f"Search time: {search_time:.3f}s (should be very fast)")
        print(f"Instant moves: {engine.nudge_stats['instant_moves']}")
        print(f"Time saved: {engine.nudge_stats['instant_time_saved']:.3f}s")
        
        if search_time < 0.1:  # Should be nearly instant
            print("✅ Instant nudge working - very fast search")
        else:
            print("⚠️ Search took longer than expected")
    else:
        print("Starting position does not have instant nudge move")
        
        # Test normal search for comparison
        start_time = time.time()
        best_move = engine.search(board, time_limit=1.0)
        search_time = time.time() - start_time
        
        print(f"Normal search time: {search_time:.3f}s")
        print(f"Best move: {best_move}")
    
    print()

def test_nudge_threshold_tuning():
    """Test different threshold values"""
    print("=== Testing Nudge Threshold Tuning ===")
    engine = V7P3REngine()
    
    board = chess.Board()
    
    # Test with different threshold values
    thresholds = [5.0, 10.0, 15.0, 20.0]
    
    for threshold in thresholds:
        engine.nudge_instant_config['confidence_threshold'] = threshold
        instant_move = engine._check_instant_nudge_move(board)
        
        print(f"Threshold {threshold}: {'✅ Instant move' if instant_move else '❌ No instant move'}")
    
    # Reset to default
    engine.nudge_instant_config['confidence_threshold'] = 12.0
    print()

def test_instant_nudge_statistics():
    """Test instant nudge statistics tracking"""
    print("=== Testing Instant Nudge Statistics ===")
    engine = V7P3REngine()
    
    # Run several searches to accumulate stats
    board = chess.Board()
    
    initial_instant_moves = engine.nudge_stats['instant_moves']
    initial_time_saved = engine.nudge_stats['instant_time_saved']
    
    for i in range(5):
        engine.search(board, time_limit=1.0)
        
        # Make a move to change position
        legal_moves = list(board.legal_moves)
        if legal_moves:
            board.push(legal_moves[0])
    
    final_instant_moves = engine.nudge_stats['instant_moves']
    final_time_saved = engine.nudge_stats['instant_time_saved']
    
    print(f"Instant moves: {initial_instant_moves} → {final_instant_moves}")
    print(f"Time saved: {initial_time_saved:.3f}s → {final_time_saved:.3f}s")
    
    if final_instant_moves > initial_instant_moves:
        print("✅ Instant nudge statistics tracking working")
    else:
        print("⚠️ No instant moves detected in test")
    
    print()

def main():
    """Run all instant nudge tests"""
    print("V7P3R v11 Phase 2 Enhancement - Instant Nudge Move System")
    print("=" * 60)
    
    test_instant_nudge_configuration()
    test_instant_nudge_detection()
    test_search_with_instant_nudges()
    test_nudge_threshold_tuning()
    test_instant_nudge_statistics()
    
    print("=" * 60)
    print("Instant nudge move system testing complete!")

if __name__ == "__main__":
    main()
