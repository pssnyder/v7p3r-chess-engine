#!/usr/bin/env python3
"""
V12.3 Time Management Test - Verify improved depth reaching
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

def test_depth_performance():
    """Test if the engine reaches better depths with v12.3 improvements"""
    
    print("=" * 60)
    print("V7P3R v12.3 - TIME MANAGEMENT & DEPTH TEST")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    test_positions = [
        {
            "name": "Starting Position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "target_depth": 5
        },
        {
            "name": "Complex Middle Game",
            "fen": "r2qkb1r/ppp2ppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 6",
            "target_depth": 4
        },
        {
            "name": "Simple Endgame",
            "fen": "8/8/8/4k3/4K3/8/8/8 w - - 0 1",
            "target_depth": 6
        }
    ]
    
    time_limits = [5.0, 10.0, 15.0]  # Test different time allocations
    
    for pos in test_positions:
        print(f"\nPosition: {pos['name']}")
        print(f"FEN: {pos['fen']}")
        print(f"Target Depth: {pos['target_depth']}")
        print("-" * 40)
        
        board = chess.Board(pos['fen'])
        
        for time_limit in time_limits:
            print(f"\nTime limit: {time_limit}s")
            
            # Reset engine state
            engine = V7P3REngine()
            
            # Capture output to see depth reached
            start_time = time.perf_counter()
            
            try:
                best_move = engine.search(board, time_limit=time_limit)
                elapsed = time.perf_counter() - start_time
                
                # The search method prints depth info, so we can't capture it easily
                # But we can check if move is reasonable
                print(f"Best move: {best_move}")
                print(f"Actual time used: {elapsed:.2f}s")
                print(f"Nodes searched: {engine.nodes_searched}")
                
                if engine.nodes_searched > 0:
                    nps = int(engine.nodes_searched / max(elapsed, 0.001))
                    print(f"NPS: {nps:,}")
                
                # Test castling preference if applicable
                if engine._is_castling_move(board, best_move):
                    print("✅ Engine chose castling move!")
                
            except Exception as e:
                print(f"❌ Error: {e}")

def test_time_allocation():
    """Test the new adaptive time allocation system"""
    
    print("\n" + "=" * 60)
    print("ADAPTIVE TIME ALLOCATION TEST")
    print("=" * 60)
    
    engine = V7P3REngine()
    base_time = 30.0  # 30 seconds base
    
    test_scenarios = [
        {
            "name": "Opening Position",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "expected": "Conservative time usage"
        },
        {
            "name": "Complex Middle Game",
            "fen": "r2qkb1r/ppp2ppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 6",
            "expected": "Aggressive time usage"
        },
        {
            "name": "King in Check",
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 0 2",
            "expected": "Extended time for check"
        },
        {
            "name": "Few Legal Moves",
            "fen": "8/8/8/8/8/8/8/K1k5 w - - 0 1",
            "expected": "Less time with few options"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Expected: {scenario['expected']}")
        
        board = chess.Board(scenario['fen'])
        target_time, max_time = engine._calculate_adaptive_time_allocation(board, base_time)
        
        print(f"Base time: {base_time:.1f}s")
        print(f"Target time: {target_time:.1f}s ({target_time/base_time*100:.1f}%)")
        print(f"Max time: {max_time:.1f}s ({max_time/base_time*100:.1f}%)")
        
        # Additional info
        moves_played = len(board.move_stack)
        legal_moves = len(list(board.legal_moves))
        in_check = board.is_check()
        
        print(f"Moves played: {moves_played}")
        print(f"Legal moves: {legal_moves}")
        print(f"In check: {in_check}")

if __name__ == "__main__":
    test_depth_performance()
    test_time_allocation()
    
    print("\n" + "=" * 60)
    print("TIME MANAGEMENT TEST COMPLETE")
    print("=" * 60)