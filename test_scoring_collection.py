#!/usr/bin/env python3
"""
Test to see what scoring data is being collected
"""
import chess
import sys
import os

# Add the v7p3r_engine path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'v7p3r_engine'))

from v7p3r_engine.v7p3r_play import v7p3rEngine
from enhanced_scoring_collector import EnhancedScoringCollector

def test_scoring_collection():
    print("Testing Enhanced Scoring Collection")
    print("=" * 40)
    
    # Create test board
    board = chess.Board()
    
    # Create V7P3R engine
    engine = v7p3rEngine()
    
    # Create scoring collector
    collector = EnhancedScoringCollector()
    
    # Test detailed scoring
    try:
        detailed_scores = collector.collect_detailed_scoring(
            engine.scoring_calculator, board, chess.WHITE
        )
        
        print("Detailed scoring collected successfully:")
        non_zero_scores = {k: v for k, v in detailed_scores.items() if v != 0.0}
        print(f"Non-zero scores: {len(non_zero_scores)} / {len(detailed_scores)}")
        
        for score_name, score_value in non_zero_scores.items():
            print(f"  {score_name}: {score_value}")
            
        if not non_zero_scores:
            print("  All scores are zero!")
            print("  Sample scores:")
            for i, (name, value) in enumerate(detailed_scores.items()):
                if i < 5:  # Show first 5
                    print(f"    {name}: {value}")
                    
    except Exception as e:
        print(f"Error collecting detailed scores: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scoring_collection()
