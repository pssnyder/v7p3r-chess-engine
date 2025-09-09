#!/usr/bin/env python3
"""
V7P3R v11 Phase 2 - Nudge System Validation Test
Test the nudge system integration and functionality
"""

import sys
import os
import time
import chess

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def test_nudge_database_loading():
    """Test that nudge database loads correctly"""
    print("=== Testing Nudge Database Loading ===")
    engine = V7P3REngine()
    
    print(f"Nudge database loaded: {len(engine.nudge_database)} positions")
    print(f"Nudge stats initialized: {engine.nudge_stats}")
    
    # Check if we have some positions
    if len(engine.nudge_database) > 0:
        print("✅ Nudge database loaded successfully")
        
        # Show a sample position
        sample_key = list(engine.nudge_database.keys())[0]
        sample_data = engine.nudge_database[sample_key]
        print(f"Sample position: {sample_key}")
        print(f"FEN: {sample_data.get('fen', 'N/A')}")
        print(f"Moves: {len(sample_data.get('moves', {}))}")
    else:
        print("❌ Nudge database is empty or failed to load")
    
    print()

def test_position_key_generation():
    """Test position key generation"""
    print("=== Testing Position Key Generation ===")
    engine = V7P3REngine()
    
    # Test with starting position
    board = chess.Board()
    key = engine._get_position_key(board)
    print(f"Starting position key: {key}")
    
    # Test with a different position
    board.push_san("e4")
    key2 = engine._get_position_key(board)
    print(f"After e4 key: {key2}")
    
    # Keys should be different
    if key != key2:
        print("✅ Position keys are unique for different positions")
    else:
        print("❌ Position keys are not unique")
    
    print()

def test_nudge_bonus_calculation():
    """Test nudge bonus calculation for known positions"""
    print("=== Testing Nudge Bonus Calculation ===")
    engine = V7P3REngine()
    
    if len(engine.nudge_database) == 0:
        print("❌ No nudge database loaded, skipping bonus test")
        return
    
    # Try to find a position in the database and test it
    found_position = False
    for position_key, position_data in engine.nudge_database.items():
        try:
            fen = position_data.get('fen')
            if fen:
                board = chess.Board(fen)
                moves = position_data.get('moves', {})
                
                print(f"Testing position: {fen}")
                print(f"Available nudge moves: {list(moves.keys())}")
                
                # Test nudge bonus for each available move
                for move_uci in moves.keys():
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            bonus = engine._get_nudge_bonus(board, move)
                            move_data = moves[move_uci]
                            print(f"Move {move_uci}: bonus={bonus:.1f}, freq={move_data.get('frequency', 0)}, eval={move_data.get('eval', 0.0):.2f}")
                            found_position = True
                            break
                    except:
                        continue
                
                if found_position:
                    break
        except Exception as e:
            continue
    
    if found_position:
        print("✅ Nudge bonus calculation working")
    else:
        print("❌ Could not test nudge bonus calculation")
    
    print()

def test_move_ordering_with_nudges():
    """Test that move ordering includes nudge bonuses"""
    print("=== Testing Move Ordering with Nudges ===")
    engine = V7P3REngine()
    
    if len(engine.nudge_database) == 0:
        print("❌ No nudge database loaded, skipping move ordering test")
        return
    
    # Try to find a position with nudge moves
    found_nudge_position = False
    for position_key, position_data in engine.nudge_database.items():
        try:
            fen = position_data.get('fen')
            if fen:
                board = chess.Board(fen)
                moves = position_data.get('moves', {})
                
                # Check if any nudge moves are legal in this position
                legal_nudge_moves = []
                for move_uci in moves.keys():
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            legal_nudge_moves.append(move)
                    except:
                        continue
                
                if legal_nudge_moves:
                    print(f"Testing move ordering for position: {fen}")
                    print(f"Legal nudge moves: {[m.uci() for m in legal_nudge_moves]}")
                    
                    # Get move ordering
                    legal_moves = list(board.legal_moves)
                    ordered_moves = engine._order_moves_advanced(board, legal_moves, 3)
                    
                    print(f"Move ordering (first 5): {[m.uci() for m in ordered_moves[:5]]}")
                    
                    # Check if nudge moves appear early in ordering
                    for i, move in enumerate(ordered_moves[:5]):
                        if move in legal_nudge_moves:
                            print(f"✅ Nudge move {move.uci()} appears at position {i+1}")
                            found_nudge_position = True
                            break
                    
                    break
        except Exception as e:
            continue
    
    if not found_nudge_position:
        print("❌ Could not find position with legal nudge moves for testing")
    
    print()

def test_search_with_nudges():
    """Test search with nudge system active"""
    print("=== Testing Search with Nudge System ===")
    engine = V7P3REngine()
    
    # Test on starting position
    board = chess.Board()
    print("Testing search on starting position...")
    
    start_time = time.time()
    best_move = engine.search(board, time_limit=1.0)
    search_time = time.time() - start_time
    
    print(f"Best move: {best_move}")
    print(f"Search time: {search_time:.3f}s")
    print(f"Nodes searched: {engine.nodes_searched}")
    print(f"Nudge stats: {engine.nudge_stats}")
    print(f"Search stats - nudge hits: {engine.search_stats.get('nudge_hits', 0)}")
    print(f"Search stats - nudge positions: {engine.search_stats.get('nudge_positions', 0)}")
    
    if best_move != chess.Move.null():
        print("✅ Search completed successfully with nudge system")
    else:
        print("❌ Search failed")
    
    print()

def test_nudge_statistics():
    """Test nudge statistics tracking"""
    print("=== Testing Nudge Statistics ===")
    engine = V7P3REngine()
    
    # Run a few searches to accumulate statistics
    board = chess.Board()
    for i in range(3):
        engine.search(board, time_limit=0.5)
        # Make a random move to change position
        legal_moves = list(board.legal_moves)
        if legal_moves:
            board.push(legal_moves[0])
    
    print("After 3 short searches:")
    print(f"Nudge database stats: {engine.nudge_stats}")
    print(f"Search stats nudge info: nudge_hits={engine.search_stats.get('nudge_hits', 0)}, nudge_positions={engine.search_stats.get('nudge_positions', 0)}")
    
    total_lookups = engine.nudge_stats['hits'] + engine.nudge_stats['misses']
    if total_lookups > 0:
        hit_rate = (engine.nudge_stats['hits'] / total_lookups) * 100
        print(f"Nudge hit rate: {hit_rate:.1f}%")
        print("✅ Nudge statistics tracking working")
    else:
        print("⚠️ No nudge lookups performed")
    
    print()

def main():
    """Run all nudge system tests"""
    print("V7P3R v11 Phase 2 - Nudge System Validation")
    print("=" * 50)
    
    test_nudge_database_loading()
    test_position_key_generation()
    test_nudge_bonus_calculation()
    test_move_ordering_with_nudges()
    test_search_with_nudges()
    test_nudge_statistics()
    
    print("=" * 50)
    print("Nudge system validation complete!")

if __name__ == "__main__":
    main()
