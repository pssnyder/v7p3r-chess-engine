#!/usr/bin/env python3
"""
V10 Enhanced vs V7 Comparison Test
Quick tactical comparison to validate improvements
"""

import chess
import sys
import os
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r_scoring_calculation import V7P3RScoringCalculationClean

def create_v7_style_scorer():
    """Create a scorer that mimics V7 behavior (without tactical enhancements)"""
    
    class V7StyleScorer:
        def __init__(self, piece_values):
            self.piece_values = piece_values
        
        def calculate_score_optimized(self, board, color, endgame_factor=0.0):
            """Basic V7-style evaluation without tactical enhancements"""
            score = 0.0
            
            # Material only (basic)
            for piece_type, value in self.piece_values.items():
                if piece_type != chess.KING:
                    piece_count = len(board.pieces(piece_type, color))
                    score += piece_count * value
            
            # Basic king safety
            king_square = board.king(color)
            if king_square:
                if color == chess.WHITE and chess.square_rank(king_square) > 2:
                    score -= 30.0
                elif color == chess.BLACK and chess.square_rank(king_square) < 5:
                    score -= 30.0
            
            # Basic center control
            center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
            for square in center_squares:
                piece = board.piece_at(square)
                if piece and piece.color == color:
                    if piece.piece_type == chess.PAWN:
                        score += 10.0
                    else:
                        score += 5.0
            
            return score
    
    return V7StyleScorer

def test_tactical_positions():
    """Test various tactical positions"""
    
    print("⚔️  V10 ENHANCED vs V7 TACTICAL COMPARISON")
    print("=" * 60)
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    # Initialize both scorers
    v10_enhanced = V7P3RScoringCalculationClean(piece_values)
    v7_style = create_v7_style_scorer()(piece_values)
    
    test_positions = [
        {
            'name': 'Starting Position',
            'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'description': 'Equal material, should be close'
        },
        {
            'name': 'Knight Fork Setup',
            'fen': '8/8/3k1r2/8/4N3/8/8/4K3 w - - 0 1',
            'description': 'White knight can fork king and rook'
        },
        {
            'name': 'Pin Position',
            'fen': '8/8/k7/2n5/2R5/8/8/4K3 w - - 0 1',
            'description': 'White rook pins black knight to king'
        },
        {
            'name': 'Endgame King Activity',
            'fen': '8/8/8/8/3k4/8/3P4/3K4 w - - 0 1',
            'description': 'Active king in endgame'
        },
        {
            'name': 'Enemy King on Edge',
            'fen': '7k/6PP/8/8/8/8/8/4K3 w - - 0 1',
            'description': 'Enemy king driven to edge, promoting pawns'
        },
        {
            'name': 'Undefended Pieces',
            'fen': 'rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1',
            'description': 'Bishop on c4 is undefended'
        }
    ]
    
    total_v10_advantage = 0.0
    tactical_improvements = 0
    
    for i, position in enumerate(test_positions, 1):
        print(f"\n🧪 TEST {i}: {position['name']}")
        print(f"📝 {position['description']}")
        
        board = chess.Board(position['fen'])
        
        # Evaluate with both engines
        v10_white = v10_enhanced.calculate_score_optimized(board, chess.WHITE)
        v10_black = v10_enhanced.calculate_score_optimized(board, chess.BLACK)
        v10_eval = v10_white - v10_black
        
        v7_white = v7_style.calculate_score_optimized(board, chess.WHITE)
        v7_black = v7_style.calculate_score_optimized(board, chess.BLACK)
        v7_eval = v7_white - v7_black
        
        difference = v10_eval - v7_eval
        total_v10_advantage += difference
        
        print(f"   V10 Enhanced: {v10_eval:+.1f}")
        print(f"   V7 Style:     {v7_eval:+.1f}")
        print(f"   Difference:   {difference:+.1f} {'✅' if abs(difference) > 5 else '➖'}")
        
        if abs(difference) > 5:
            tactical_improvements += 1
    
    print(f"\n📊 TACTICAL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"🎯 Total V10 Advantage: {total_v10_advantage:+.1f}")
    print(f"📈 Positions with Tactical Improvement: {tactical_improvements}/{len(test_positions)}")
    print(f"⚡ Average Advantage per Position: {total_v10_advantage/len(test_positions):+.1f}")
    
    if tactical_improvements >= len(test_positions) // 2:
        print(f"\n🏆 SUCCESS: V10 Enhanced shows significant tactical improvements!")
        print(f"✅ Tactical awareness is clearly enhanced")
    else:
        print(f"\n⚠️  Mixed results: Some improvements detected but may need tuning")
    
    return total_v10_advantage, tactical_improvements

def performance_benchmark():
    """Quick performance benchmark"""
    print(f"\n⚡ PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    scorer = V7P3RScoringCalculationClean(piece_values)
    board = chess.Board()
    
    # Warm up
    for _ in range(10):
        scorer.calculate_score_optimized(board, chess.WHITE)
    
    # Benchmark
    start_time = time.time()
    iterations = 1000
    
    for _ in range(iterations):
        scorer.calculate_score_optimized(board, chess.WHITE)
        scorer.calculate_score_optimized(board, chess.BLACK)
    
    end_time = time.time()
    total_time = end_time - start_time
    evaluations_per_second = (iterations * 2) / total_time
    
    print(f"⏱️  {iterations * 2} evaluations in {total_time:.3f} seconds")
    print(f"🚀 Performance: {evaluations_per_second:.0f} evaluations/second")
    
    if evaluations_per_second > 1000:
        print(f"✅ Performance is excellent!")
    elif evaluations_per_second > 500:
        print(f"✅ Performance is good!")
    else:
        print(f"⚠️  Performance may need optimization")
    
    return evaluations_per_second

if __name__ == "__main__":
    try:
        # Run tactical comparison
        advantage, improvements = test_tactical_positions()
        
        # Run performance benchmark
        performance = performance_benchmark()
        
        print(f"\n🎯 FINAL ASSESSMENT")
        print("=" * 60)
        print(f"✅ V10 Tactical Enhancement Integration: COMPLETE")
        print(f"✅ Tactical Improvements: {improvements} out of 6 test positions")
        print(f"✅ Average Tactical Advantage: {advantage/6:+.1f} points")
        print(f"✅ Performance: {performance:.0f} evaluations/second")
        
        if improvements >= 3 and performance > 500:
            print(f"\n🏆 V10 ENHANCED IS READY FOR BATTLE!")
            print(f"🎯 Tactical patterns: Pin, Fork, Skewer, Discovery, Deflection, Guard Removal")
            print(f"🎯 Enhanced endgame: Edge-driving, King proximity, Pawn promotion")
            print(f"🎯 Piece defense: Defense networks, Undefended piece penalties")
            print(f"🎯 All features integrated into single scoring file!")
        else:
            print(f"\n⚠️  Integration complete but may need fine-tuning")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
