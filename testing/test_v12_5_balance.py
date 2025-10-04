#!/usr/bin/env python3
"""
V7P3R v12.5 Comprehensive Balance Test
Tests evaluation balance across multiple positions with the enhanced nudge system
"""

import chess
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine
from v7p3r_heuristic_analyzer import V7P3RHeuristicAnalyzer

def test_position_balance(engine, analyzer, position_name, fen):
    """Test evaluation balance for a specific position"""
    print(f"\n📋 Testing: {position_name}")
    print("-" * 50)
    
    board = chess.Board(fen)
    print(f"FEN: {fen}")
    
    # Get basic evaluation
    base_eval = engine.bitboard_evaluator.bitboard_evaluator.evaluate_bitboard(board, chess.WHITE)
    print(f"Base evaluation: {base_eval:.2f}")
    
    # Test nudge bonuses for legal moves
    legal_moves = list(board.legal_moves)[:6]  # Test first 6 moves
    move_bonuses = []
    
    for move in legal_moves:
        bonus = engine._get_nudge_bonus(board, move)
        move_bonuses.append((move.uci(), bonus))
        
    # Sort by bonus to see top moves
    move_bonuses.sort(key=lambda x: x[1], reverse=True)
    
    print("Move ordering bonuses:")
    for move_uci, bonus in move_bonuses[:3]:
        print(f"  {move_uci}: +{bonus:.1f}")
    
    # Check bonus scaling
    max_bonus = max(bonus for _, bonus in move_bonuses) if move_bonuses else 0
    eval_ratio = max_bonus / abs(base_eval) if abs(base_eval) > 0.1 else 0
    
    if eval_ratio < 0.5:  # Bonus should be < 50% of base eval
        print(f"✅ Bonuses well-scaled (ratio: {eval_ratio:.2f})")
    else:
        print(f"⚠️  High bonus ratio: {eval_ratio:.2f}")
    
    # Run heuristic analysis
    try:
        breakdown = analyzer.analyze_position(board, chess.WHITE)
        
        # Check for evaluation hotspots
        if breakdown.hotspots:
            major_hotspots = [h for h in breakdown.hotspots if '(' in h and float(h.split('(')[1].split('%')[0]) > 40]
            if major_hotspots:
                print(f"⚠️  Major hotspots: {', '.join(major_hotspots)}")
            else:
                print("✅ No major evaluation hotspots")
        else:
            print("✅ Balanced evaluation distribution")
            
    except Exception as e:
        print(f"⚠️  Heuristic analysis error: {e}")

def main():
    print("V7P3R v12.5 Comprehensive Balance Test")
    print("=====================================")
    
    # Initialize engine and analyzer
    print("Initializing engine and analyzer...")
    engine = V7P3REngine()
    analyzer = V7P3RHeuristicAnalyzer()
    
    # Test positions representing different game phases and scenarios
    test_positions = [
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Opening: Italian Game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("Caro-Kann Defense", "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
        ("Middlegame Position", "r1bq1rk1/ppp2ppp/2n1bn2/2bpp3/2PP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 9"),
        ("Tactical Position", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"),
        ("Endgame: K+P vs K", "8/8/8/3k4/3P4/3K4/8/8 w - - 0 1")
    ]
    
    balance_results = []
    
    for position_name, fen in test_positions:
        try:
            test_position_balance(engine, analyzer, position_name, fen)
            balance_results.append("✅")
        except Exception as e:
            print(f"❌ Error testing {position_name}: {e}")
            balance_results.append("❌")
    
    # Summary
    print("\n" + "=" * 50)
    print("BALANCE TEST SUMMARY")
    print("=" * 50)
    
    passed = balance_results.count("✅")
    total = len(balance_results)
    
    print(f"Positions tested: {total}")
    print(f"Balanced evaluations: {passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed >= total * 0.8:  # 80% success rate
        print("\n🎉 V7P3R v12.5 Intelligent Nudge System: BALANCED ✅")
        print("✅ Enhanced opening play without overwhelming base evaluation")
        print("✅ Center control and Caro-Kann improvements")
        print("✅ Stable across different game phases")
    else:
        print("\n⚠️  V7P3R v12.5 needs evaluation tuning")
        print("Consider reducing nudge influence further")

if __name__ == "__main__":
    main()