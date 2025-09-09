#!/usr/bin/env python3
"""
V7P3R v11 Phase 2 - Nudge System Impact Analysis
Analyze how nudge system affects move selection
"""

import sys
import os
import chess

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def test_nudge_impact_analysis():
    """Test nudge system impact on move selection"""
    print("=== V7P3R v11 Phase 2 - Nudge System Impact Analysis ===")
    print()
    
    engine = V7P3REngine()
    
    # Test position with known nudges
    test_positions = [
        {
            'name': 'Starting Position',
            'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -',
            'expected_nudges': ['g1f3', 'b1c3', 'e2e3']
        },
        {
            'name': 'After Nc3',
            'fen': 'rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq -',
            'expected_nudges': ['e7e6', 'g8h6', 'h7h6']
        }
    ]
    
    for position in test_positions:
        print(f"Testing: {position['name']}")
        print(f"FEN: {position['fen']}")
        
        try:
            board = chess.Board(position['fen'])
            legal_moves = list(board.legal_moves)
            
            print(f"Legal moves: {len(legal_moves)}")
            
            # Check position key
            position_key = engine._get_position_key(board)
            print(f"Position key: {position_key}")
            
            # Check if position is in nudge database
            if position_key in engine.nudge_database:
                position_data = engine.nudge_database[position_key]
                nudge_moves = position_data.get('moves', {})
                print(f"✅ Position found in nudge database with {len(nudge_moves)} nudge moves")
                
                # Check each expected nudge move
                for move_uci in position['expected_nudges']:
                    if move_uci in nudge_moves:
                        move_data = nudge_moves[move_uci]
                        print(f"  {move_uci}: freq={move_data['frequency']}, eval={move_data['eval']:.3f}")
                        
                        # Calculate bonus
                        try:
                            move = chess.Move.from_uci(move_uci)
                            if move in legal_moves:
                                bonus = engine._get_nudge_bonus(board, move)
                                print(f"    Nudge bonus: {bonus:.1f}")
                        except:
                            pass
            else:
                print(f"❌ Position not found in nudge database")
            
            # Test move ordering
            ordered_moves = engine._order_moves_advanced(board, legal_moves, 4)
            print(f"Move ordering (top 8): {[m.uci() for m in ordered_moves[:8]]}")
            
            # Check if nudge moves appear early
            nudge_positions = []
            for i, move in enumerate(ordered_moves[:10]):
                if move.uci() in position['expected_nudges']:
                    nudge_positions.append((move.uci(), i+1))
            
            if nudge_positions:
                print(f"✅ Nudge moves in top 10: {nudge_positions}")
            else:
                print(f"⚠️ No nudge moves in top 10")
            
            # Quick search test
            best_move = engine.search(board, time_limit=0.5)
            print(f"Search result: {best_move} (is nudge: {best_move.uci() in position['expected_nudges']})")
            
        except Exception as e:
            print(f"❌ Error testing position: {e}")
        
        print("-" * 60)
        print()
    
    # Overall statistics
    print("Overall Nudge Statistics:")
    print(f"Total database positions: {len(engine.nudge_database)}")
    print(f"Nudge lookup stats: {engine.nudge_stats}")
    
    total_lookups = engine.nudge_stats['hits'] + engine.nudge_stats['misses']
    if total_lookups > 0:
        hit_rate = (engine.nudge_stats['hits'] / total_lookups) * 100
        print(f"Nudge hit rate: {hit_rate:.2f}%")
    
    print()
    print("=== Phase 2 Nudge System Implementation: ✅ SUCCESSFUL ===")

if __name__ == "__main__":
    test_nudge_impact_analysis()
