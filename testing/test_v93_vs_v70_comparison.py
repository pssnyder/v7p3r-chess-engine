#!/usr/bin/env python3
"""
V9.3 vs V7.0 Head-to-Head Comparison Test
Systematic comparison to identify weaknesses and guide v9.4 development
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
import json
from datetime import datetime
from v7p3r import V7P3RCleanEngine

# Import the appropriate scoring calculations
try:
    from v7p3r_scoring_calculation_v93 import V7P3RScoringCalculationV93
    v93_available = True
except ImportError:
    v93_available = False
    print("Warning: v9.3 scoring calculation not available")

try:
    from v7p3r_scoring_calculation import V7P3RScoringCalculation
    v70_available = True
except ImportError:
    v70_available = False
    print("Warning: v7.0 scoring calculation not available")

# Critical test positions from tournament history
TEST_POSITIONS = [
    {
        "name": "Opening Control",
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "importance": "high",
        "category": "opening"
    },
    {
        "name": "Knight Development",
        "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "importance": "high", 
        "category": "development"
    },
    {
        "name": "Tactical Fork Setup",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "importance": "critical",
        "category": "tactics"
    },
    {
        "name": "Material Imbalance",
        "fen": "r3k2r/8/8/8/8/8/8/4Q2K w kq - 0 1",
        "importance": "high",
        "category": "material"
    },
    {
        "name": "Endgame King Activity",
        "fen": "8/8/8/8/8/3k4/3P4/3K4 w - - 0 1",
        "importance": "medium",
        "category": "endgame"
    },
    {
        "name": "Pawn Structure",
        "fen": "r1bqkb1r/pp2pppp/2np1n2/8/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 5",
        "importance": "medium",
        "category": "structure"
    },
    {
        "name": "King Safety",
        "fen": "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 6",
        "importance": "critical",
        "category": "safety"
    },
    {
        "name": "Pin Pressure",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4",
        "importance": "high",
        "category": "tactics"
    }
]

def compare_engines_on_position(position_data, time_limit=2.0):
    """Compare v9.3 and v7.0 on a specific position"""
    print(f"\nTesting: {position_data['name']} ({position_data['category']})")
    print(f"Importance: {position_data['importance']}")
    
    board = chess.Board(position_data['fen'])
    results = {}
    
    # Test v9.3 
    print("  V9.3 Analysis:")
    v93_engine = V7P3RCleanEngine()
    
    start_time = time.time()
    try:
        v93_move = v93_engine.search(board, time_limit=time_limit)
        v93_time = time.time() - start_time
        v93_eval = v93_engine._evaluate_position_deterministic(board)
        
        print(f"    Move: {v93_move}")
        print(f"    Eval: {v93_eval:.4f}")
        print(f"    Time: {v93_time:.3f}s")
        
        results['v93'] = {
            'move': str(v93_move) if v93_move else None,
            'eval': v93_eval,
            'time': v93_time
        }
    except Exception as e:
        print(f"    Error: {e}")
        results['v93'] = {'error': str(e)}
    
    # For now, we'll create a placeholder for v7.0 comparison
    # TODO: Add actual v7.0 engine comparison once available
    print("  V7.0 Analysis:")
    print("    [V7.0 comparison will be added in next iteration]")
    
    results['position'] = position_data
    return results

def run_comprehensive_comparison():
    """Run full comparison suite"""
    print("V7P3R v9.3 vs v7.0 Comprehensive Comparison")
    print("=" * 60)
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v9.3 vs v7.0',
        'positions': []
    }
    
    critical_wins = 0
    high_wins = 0
    total_positions = len(TEST_POSITIONS)
    
    for position_data in TEST_POSITIONS:
        result = compare_engines_on_position(position_data)
        all_results['positions'].append(result)
        
        # Track performance on critical positions
        if position_data['importance'] == 'critical':
            # For now, assume we're analyzing the moves
            critical_wins += 1  # Placeholder
        elif position_data['importance'] == 'high':
            high_wins += 1  # Placeholder
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"v93_vs_v70_comparison_{timestamp}.json"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nComparison Results Summary:")
    print(f"Total positions tested: {total_positions}")
    print(f"Critical positions: {sum(1 for p in TEST_POSITIONS if p['importance'] == 'critical')}")
    print(f"High importance: {sum(1 for p in TEST_POSITIONS if p['importance'] == 'high')}")
    print(f"Results saved to: {filename}")
    
    return all_results

def analyze_move_quality(position_data):
    """Analyze the quality of moves chosen by v9.3"""
    print(f"\nDetailed Move Analysis: {position_data['name']}")
    board = chess.Board(position_data['fen'])
    
    # Get all legal moves
    legal_moves = list(board.legal_moves)
    print(f"Legal moves: {len(legal_moves)}")
    
    # Test v9.3's evaluation of top moves
    v93_engine = V7P3RCleanEngine()
    
    move_evaluations = []
    for move in legal_moves[:8]:  # Test top 8 moves
        test_board = board.copy()
        test_board.push(move)
        eval_score = v93_engine._evaluate_position_deterministic(test_board)
        move_evaluations.append((move, eval_score))
    
    # Sort by evaluation
    move_evaluations.sort(key=lambda x: x[1], reverse=True)
    
    print("Top moves by v9.3 evaluation:")
    for i, (move, score) in enumerate(move_evaluations[:5]):
        print(f"  {i+1}. {move}: {score:.4f}")
    
    return move_evaluations

def main():
    """Main comparison function"""
    if not v93_available:
        print("❌ v9.3 scoring calculation not available!")
        return False
    
    print("Starting v9.3 vs v7.0 comparison analysis...")
    
    # Run comprehensive comparison
    results = run_comprehensive_comparison()
    
    # Analyze a few critical positions in detail
    critical_positions = [p for p in TEST_POSITIONS if p['importance'] == 'critical']
    
    print("\n" + "="*60)
    print("DETAILED ANALYSIS OF CRITICAL POSITIONS")
    print("="*60)
    
    for pos in critical_positions:
        analyze_move_quality(pos)
    
    print("\n🎯 Next Steps for v9.4 Development:")
    print("1. Analyze move choices vs chess principles")
    print("2. Compare evaluations with known good positions")
    print("3. Identify heuristic adjustments needed")
    print("4. Test incremental improvements")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
