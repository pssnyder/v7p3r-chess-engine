#!/usr/bin/env python3
"""
V9.4-Alpha vs V7.0 Head-to-Head Test
Direct comparison to validate our simplified approach beats v7.0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
import json
from datetime import datetime

# Import both scoring systems
try:
    from v7p3r_scoring_calculation_v94_alpha import V7P3RScoringCalculationV94Alpha
    v94_available = True
except ImportError:
    v94_available = False
    print("Warning: v9.4-alpha scoring calculation not available")

try:
    from v7p3r_scoring_calculation import V7P3RScoringCalculationClean
    v70_available = True
except ImportError:
    v70_available = False
    print("Warning: v7.0 scoring calculation not available")

# Standard piece values for fair comparison
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320, 
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

# Test positions that expose v7.0's weaknesses
TACTICAL_TEST_POSITIONS = [
    {
        "name": "Knight Fork Opportunity",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "v70_weakness": "Misses tactical opportunities",
        "v94_advantage": "Should detect knight fork potential"
    },
    {
        "name": "Hanging Piece Detection",
        "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "v70_weakness": "Doesn't detect undefended pieces",
        "v94_advantage": "Should penalize hanging pieces"
    },
    {
        "name": "Center Control Battle",
        "fen": "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",
        "v70_weakness": "Basic center evaluation",
        "v94_advantage": "Enhanced center control assessment"
    },
    {
        "name": "Pawn Promotion Threat",
        "fen": "8/1P6/8/8/8/8/k7/K7 w - - 0 1",
        "v70_weakness": "Weak endgame evaluation",
        "v94_advantage": "Should recognize promotion power"
    },
    {
        "name": "King Exposure Test",
        "fen": "r3k2r/pbppnppp/1bn5/1B6/1Q6/3P1N2/PPP2PPP/RNB1K2R b KQkq - 0 8",
        "v70_weakness": "Basic king safety",
        "v94_advantage": "Enhanced king safety with attack detection"
    }
]

def compare_evaluations(position_data):
    """Compare v9.4-alpha vs v7.0 evaluation on a position"""
    print(f"\n{'='*60}")
    print(f"Position: {position_data['name']}")
    print(f"FEN: {position_data['fen']}")
    print(f"V7.0 Weakness: {position_data['v70_weakness']}")
    print(f"V9.4 Advantage: {position_data['v94_advantage']}")
    print(f"{'='*60}")
    
    board = chess.Board(position_data['fen'])
    
    results = {
        'position': position_data,
        'evaluations': {}
    }
    
    # Test v7.0 evaluation
    if v70_available:
        print("V7.0 Evaluation:")
        v70_calc = V7P3RScoringCalculationClean(PIECE_VALUES)
        
        v70_white = v70_calc.calculate_score_optimized(board, chess.WHITE)
        v70_black = v70_calc.calculate_score_optimized(board, chess.BLACK)
        v70_eval = v70_white - v70_black
        
        print(f"  White: {v70_white:.2f}")
        print(f"  Black: {v70_black:.2f}")
        print(f"  Evaluation: {v70_eval:.2f}")
        
        results['evaluations']['v70'] = {
            'white': v70_white,
            'black': v70_black,
            'eval': v70_eval
        }
    
    # Test v9.4-alpha evaluation
    if v94_available:
        print("V9.4-Alpha Evaluation:")
        v94_calc = V7P3RScoringCalculationV94Alpha(PIECE_VALUES)
        
        v94_white = v94_calc.calculate_score_optimized(board, chess.WHITE)
        v94_black = v94_calc.calculate_score_optimized(board, chess.BLACK)
        v94_eval = v94_white - v94_black
        
        print(f"  White: {v94_white:.2f}")
        print(f"  Black: {v94_black:.2f}")
        print(f"  Evaluation: {v94_eval:.2f}")
        
        results['evaluations']['v94'] = {
            'white': v94_white,
            'black': v94_black,
            'eval': v94_eval
        }
    
    # Compare the evaluations
    if v70_available and v94_available:
        eval_diff = v94_eval - v70_eval
        print(f"\nEvaluation Difference (v9.4 - v7.0): {eval_diff:.2f}")
        
        if abs(eval_diff) > 5.0:
            if eval_diff > 0:
                print("✓ V9.4 sees this position as significantly better for White")
            else:
                print("✓ V9.4 sees this position as significantly better for Black")
        else:
            print("≈ Similar evaluation between engines")
        
        results['difference'] = eval_diff
    
    return results

def analyze_component_differences(position_data):
    """Break down evaluation differences by component"""
    if not (v70_available and v94_available):
        return
        
    print(f"\nComponent Analysis for {position_data['name']}:")
    board = chess.Board(position_data['fen'])
    
    v70_calc = V7P3RScoringCalculationClean(PIECE_VALUES)
    v94_calc = V7P3RScoringCalculationV94Alpha(PIECE_VALUES)
    
    # Material should be identical
    v70_material = v70_calc._material_score(board, chess.WHITE)
    v94_material = v94_calc._material_score(board, chess.WHITE)
    print(f"  Material (White) - V7.0: {v70_material:.2f}, V9.4: {v94_material:.2f}")
    
    # King safety comparison
    v70_king = v70_calc._king_safety(board, chess.WHITE)
    v94_king = v94_calc._king_safety_enhanced(board, chess.WHITE)
    print(f"  King Safety (White) - V7.0: {v70_king:.2f}, V9.4: {v94_king:.2f}")
    
    # Development comparison
    v70_dev = v70_calc._piece_development(board, chess.WHITE)
    v94_dev = v94_calc._development_focused(board, chess.WHITE)
    print(f"  Development (White) - V7.0: {v70_dev:.2f}, V9.4: {v94_dev:.2f}")
    
    # Tactical component (v9.4 only)
    v94_tactics = v94_calc._basic_tactics(board, chess.WHITE)
    print(f"  Tactics (White) - V7.0: 0.00, V9.4: {v94_tactics:.2f}")

def run_head_to_head_analysis():
    """Run comprehensive head-to-head analysis"""
    print("V7P3R v9.4-Alpha vs v7.0 Head-to-Head Analysis")
    print("=" * 80)
    
    if not v70_available:
        print("❌ V7.0 scoring not available - cannot run comparison")
        return False
        
    if not v94_available:
        print("❌ V9.4-alpha scoring not available - cannot run comparison")
        return False
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'comparison': 'v9.4-alpha vs v7.0',
        'positions': []
    }
    
    v94_advantages = 0
    significant_differences = 0
    
    for position_data in TACTICAL_TEST_POSITIONS:
        result = compare_evaluations(position_data)
        all_results['positions'].append(result)
        
        if 'difference' in result:
            if abs(result['difference']) > 5.0:
                significant_differences += 1
                if result['difference'] > 0:
                    print("→ V9.4 advantage detected")
                    v94_advantages += 1
                else:
                    print("→ V7.0 advantage detected")
        
        # Component analysis for first few positions
        if len(all_results['positions']) <= 3:
            analyze_component_differences(position_data)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"v94_alpha_vs_v70_analysis_{timestamp}.json"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total positions tested: {len(TACTICAL_TEST_POSITIONS)}")
    print(f"Significant differences: {significant_differences}")
    print(f"V9.4 advantages: {v94_advantages}")
    print(f"Results saved to: {filename}")
    
    if v94_advantages >= len(TACTICAL_TEST_POSITIONS) // 2:
        print("\n🎉 V9.4-alpha shows promising advantages over v7.0!")
        print("✓ Ready for next development phase")
        return True
    else:
        print("\n⚠️  V9.4-alpha needs more work to consistently beat v7.0")
        print("→ Consider adjusting evaluation weights")
        return False

def main():
    """Main analysis function"""
    success = run_head_to_head_analysis()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Create engine wrapper for v9.4-alpha")
        print("2. Run actual game tests")
        print("3. Fine-tune evaluation weights")
        print("4. Progress to v9.4-beta")
    else:
        print("\n🔧 Improvement needed:")
        print("1. Analyze weak positions")
        print("2. Adjust evaluation components") 
        print("3. Re-test and iterate")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
