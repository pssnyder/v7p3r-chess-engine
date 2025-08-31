#!/usr/bin/env python3
"""
V9.4-Beta vs V7.0 Validation Test
Test the refined v9.4-beta against v7.0 after alpha improvements
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
import json
from datetime import datetime

# Import scoring systems
try:
    from v7p3r_scoring_calculation_v94_beta import V7P3RScoringCalculationV94Beta
    v94_beta_available = True
except ImportError:
    v94_beta_available = False
    print("Warning: v9.4-beta scoring calculation not available")

try:
    from v7p3r_scoring_calculation import V7P3RScoringCalculationClean
    v70_available = True
except ImportError:
    v70_available = False
    print("Warning: v7.0 scoring calculation not available")

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320, 
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

# Expanded test suite based on v9.4-alpha analysis
VALIDATION_POSITIONS = [
    {
        "name": "Opening Development",
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "target": "v94 should favor central development"
    },
    {
        "name": "Knight Fork Potential", 
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "target": "v94 should detect tactical opportunities"
    },
    {
        "name": "King Safety Critical",
        "fen": "r3k2r/pbppnppp/1bn5/1B6/1Q6/3P1N2/PPP2PPP/RNB1K2R b KQkq - 0 8", 
        "target": "v94 should properly assess king safety"
    },
    {
        "name": "Endgame Promotion",
        "fen": "8/1P6/8/8/8/8/k7/K7 w - - 0 1",
        "target": "v94 should strongly favor promotion"
    },
    {
        "name": "Piece Coordination",
        "fen": "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 6",
        "target": "v94 should reward good piece coordination"
    },
    {
        "name": "Center Control Battle",
        "fen": "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 2 3",
        "target": "v94 should recognize center importance"
    },
    {
        "name": "Tactical Pin Setup",
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5",
        "target": "v94 should see pin opportunities"
    }
]

def compare_engines_detailed(position_data):
    """Detailed comparison of v9.4-beta vs v7.0"""
    print(f"\n{'='*70}")
    print(f"Position: {position_data['name']}")
    print(f"Target: {position_data['target']}")
    print(f"FEN: {position_data['fen']}")
    print(f"{'='*70}")
    
    board = chess.Board(position_data['fen'])
    results = {}
    
    # V7.0 Analysis
    if v70_available:
        print("V7.0 Analysis:")
        v70_calc = V7P3RScoringCalculationClean(PIECE_VALUES)
        
        v70_white = v70_calc.calculate_score_optimized(board, chess.WHITE)
        v70_black = v70_calc.calculate_score_optimized(board, chess.BLACK)
        v70_eval = v70_white - v70_black
        
        print(f"  White: {v70_white:.2f}")
        print(f"  Black: {v70_black:.2f}")
        print(f"  Evaluation: {v70_eval:.2f}")
        
        results['v70'] = {
            'white': v70_white,
            'black': v70_black,
            'eval': v70_eval
        }
    
    # V9.4-Beta Analysis
    if v94_beta_available:
        print("V9.4-Beta Analysis:")
        v94_calc = V7P3RScoringCalculationV94Beta(PIECE_VALUES)
        
        v94_white = v94_calc.calculate_score_optimized(board, chess.WHITE)
        v94_black = v94_calc.calculate_score_optimized(board, chess.BLACK)
        v94_eval = v94_white - v94_black
        
        print(f"  White: {v94_white:.2f}")
        print(f"  Black: {v94_black:.2f}")
        print(f"  Evaluation: {v94_eval:.2f}")
        
        results['v94'] = {
            'white': v94_white,
            'black': v94_black,
            'eval': v94_eval
        }
        
        # Component breakdown for v9.4
        print("\nV9.4-Beta Component Breakdown (White):")
        material = v94_calc._material_score(board, chess.WHITE)
        king_safety = v94_calc._king_safety_refined(board, chess.WHITE)
        development = v94_calc._development_optimized(board, chess.WHITE)
        tactical = v94_calc._tactical_awareness(board, chess.WHITE)
        strategic = v94_calc._strategic_bonuses(board, chess.WHITE)
        
        print(f"  Material: {material:.2f}")
        print(f"  King Safety: {king_safety:.2f}")
        print(f"  Development: {development:.2f}")
        print(f"  Tactical: {tactical:.2f}")
        print(f"  Strategic: {strategic:.2f}")
        
        if v94_calc._is_endgame(board):
            endgame = v94_calc._endgame_refined(board, chess.WHITE)
            print(f"  Endgame: {endgame:.2f}")
    
    # Comparison
    if v70_available and v94_beta_available:
        eval_diff = v94_eval - v70_eval
        print(f"\nEvaluation Difference (v9.4-beta - v7.0): {eval_diff:.2f}")
        
        if abs(eval_diff) > 10.0:
            if eval_diff > 0:
                print("✓ V9.4-beta strongly favors White")
                result_assessment = "v94_advantage_white"
            else:
                print("✓ V9.4-beta strongly favors Black")
                result_assessment = "v94_advantage_black"
        elif abs(eval_diff) > 3.0:
            print("~ Moderate difference in evaluation")
            result_assessment = "moderate_difference"
        else:
            print("≈ Similar evaluations")
            result_assessment = "similar"
            
        results['comparison'] = {
            'difference': eval_diff,
            'assessment': result_assessment
        }
    
    return results

def run_beta_validation():
    """Run comprehensive v9.4-beta validation"""
    print("V7P3R v9.4-Beta vs v7.0 Validation Test")
    print("=" * 80)
    
    if not v70_available:
        print("❌ V7.0 scoring not available")
        return False
        
    if not v94_beta_available:
        print("❌ V9.4-beta scoring not available")
        return False
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v9.4-beta validation',
        'positions': []
    }
    
    v94_strong_advantages = 0
    v94_moderate_advantages = 0
    similar_evaluations = 0
    v70_advantages = 0
    
    for position_data in VALIDATION_POSITIONS:
        result = compare_engines_detailed(position_data)
        all_results['positions'].append(result)
        
        if 'comparison' in result:
            assessment = result['comparison']['assessment']
            if 'v94_advantage' in assessment:
                v94_strong_advantages += 1
            elif assessment == 'moderate_difference':
                v94_moderate_advantages += 1
            elif assessment == 'similar':
                similar_evaluations += 1
            else:
                v70_advantages += 1
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"v94_beta_validation_{timestamp}.json"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    total_positions = len(VALIDATION_POSITIONS)
    print(f"\n{'='*80}")
    print("V9.4-BETA VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total positions: {total_positions}")
    print(f"V9.4 strong advantages: {v94_strong_advantages}")
    print(f"V9.4 moderate advantages: {v94_moderate_advantages}")
    print(f"Similar evaluations: {similar_evaluations}")
    print(f"V7.0 advantages: {v70_advantages}")
    print(f"Results saved to: {filename}")
    
    # Success criteria
    v94_total_advantages = v94_strong_advantages + v94_moderate_advantages
    success_rate = v94_total_advantages / total_positions
    
    print(f"\nV9.4-beta advantage rate: {success_rate:.1%}")
    
    if success_rate >= 0.6:  # 60% advantage rate
        print("\n🎉 V9.4-beta shows strong potential to beat v7.0!")
        print("✓ Ready for actual game testing")
        print("✓ Consider progressing to v9.4-release candidate")
        return True
    elif success_rate >= 0.4:  # 40% advantage rate
        print("\n⚡ V9.4-beta shows promise but needs fine-tuning")
        print("→ Adjust evaluation weights")
        print("→ Focus on weak positions")
        return False
    else:
        print("\n❌ V9.4-beta needs significant improvements")
        print("→ Review evaluation components")
        print("→ Consider different approach")
        return False

def main():
    """Main validation function"""
    success = run_beta_validation()
    
    if success:
        print("\n🎯 Next Steps for v9.4:")
        print("1. Create v9.4 engine integration")
        print("2. Run actual chess games vs v7.0")
        print("3. Measure win/loss/draw statistics")
        print("4. Finalize v9.4 and prepare v10.0 release")
    else:
        print("\n🔧 Improvement needed:")
        print("1. Analyze specific weak positions")
        print("2. Adjust component weights")
        print("3. Re-test and iterate")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
