#!/usr/bin/env python3
"""
Simple V7P3R Move Ordering Analysis v13.x
Focused analysis of move ordering without complex board manipulation

Based on V12.6 weakness findings:
- 75% of bad moves not in top 5 -> Fix move prioritization  
- 70% tactical misses -> Better tactical move scoring
- 27.7% hanging pieces -> Add piece safety priority
"""

import chess
import sys
import os
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

class SimpleV13MoveAnalyzer:
    """Simplified V13.x move ordering analyzer"""
    
    def __init__(self):
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320, 
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
    
    def simple_categorize_move(self, board: chess.Board, move: chess.Move) -> Tuple[List[str], int]:
        """Simple move categorization with priority scoring"""
        categories = []
        priority_score = 0
        
        # Basic capture analysis
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            
            if victim and attacker:
                victim_value = self.piece_values.get(victim.piece_type, 0)
                attacker_value = self.piece_values.get(attacker.piece_type, 0)
                
                if victim_value > attacker_value:
                    categories.append("Good Capture")
                    priority_score += 500 + victim_value - attacker_value
                elif victim_value == attacker_value:
                    categories.append("Equal Capture")
                    priority_score += 200
                else:
                    categories.append("Bad Capture")
                    priority_score -= 100
        
        # Check analysis (safe because we're not modifying board)
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_check():
            categories.append("Check")
            priority_score += 300
            
            if board_copy.is_checkmate():
                categories.append("Checkmate")
                priority_score += 10000
        
        # Promotion
        if move.promotion:
            if move.promotion == chess.QUEEN:
                categories.append("Queen Promotion")
                priority_score += 800
            else:
                categories.append(f"Promotion to {chess.piece_name(move.promotion)}")
                priority_score += 400
        
        # Castling
        if board.is_castling(move):
            categories.append("Castle")
            priority_score += 150
        
        # En passant
        if board.is_en_passant(move):
            categories.append("En Passant")
            priority_score += 200
        
        # Piece development (opening)
        piece = board.piece_at(move.from_square)
        if piece:
            if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                start_rank = chess.square_rank(move.from_square)
                if ((piece.color == chess.WHITE and start_rank == 0) or 
                    (piece.color == chess.BLACK and start_rank == 7)):
                    categories.append("Piece Development")
                    priority_score += 100
        
        # If no categories, it's a quiet move
        if not categories:
            categories.append("Quiet Move")
            priority_score += 10
        
        return categories, priority_score
    
    def analyze_simple_ordering(self, board: chess.Board) -> Dict:
        """Simple move ordering analysis"""
        legal_moves = list(board.legal_moves)
        
        print(f"🔍 Simple V13.x Analysis: {len(legal_moves)} legal moves")
        
        move_analyses = []
        category_stats = defaultdict(int)
        
        for move in legal_moves:
            categories, priority_score = self.simple_categorize_move(board, move)
            
            analysis = {
                'move': move,
                'categories': categories,
                'priority_score': priority_score,
                'description': f"{move.uci()} ({board.san(move)})",
                'is_capture': board.is_capture(move),
                'is_check': "Check" in categories,
                'is_tactical': any(cat in ["Good Capture", "Check", "Queen Promotion"] for cat in categories)
            }
            
            move_analyses.append(analysis)
            
            # Update stats
            for category in categories:
                category_stats[category] += 1
        
        # Sort by priority score (highest first)
        optimal_ordering = sorted(move_analyses, key=lambda x: x['priority_score'], reverse=True)
        
        # Simple engine ordering (MVV-LVA + checks)
        engine_ordering = self._simple_engine_ordering(board, legal_moves)
        
        return {
            'move_analyses': move_analyses,
            'optimal_ordering': optimal_ordering,
            'engine_ordering': engine_ordering,
            'category_stats': dict(category_stats),
            'total_moves': len(legal_moves),
            'board': board  # Store board for later analysis
        }
    
    def _simple_engine_ordering(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """Simple engine-like move ordering"""
        def score_move(move):
            score = 0
            
            # Captures (MVV-LVA)
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    victim_value = self.piece_values.get(victim.piece_type, 0)
                    attacker_value = self.piece_values.get(attacker.piece_type, 0)
                    score += victim_value * 10 - attacker_value
            
            # Checks
            board_copy = board.copy()
            board_copy.push(move)
            if board_copy.is_check():
                score += 500
                if board_copy.is_checkmate():
                    score += 10000
            
            # Promotions
            if move.promotion == chess.QUEEN:
                score += 800
            
            return score
        
        return sorted(moves, key=score_move, reverse=True)
    
    def print_simple_analysis(self, analysis: Dict, position_name: str = "Position"):
        """Print simple V13.x analysis"""
        print(f"\n{'='*60}")
        print(f"🚀 V13.x SIMPLE MOVE ANALYSIS: {position_name}")
        print(f"{'='*60}")
        
        move_analyses = analysis['move_analyses']
        optimal_ordering = analysis['optimal_ordering']
        engine_ordering = analysis['engine_ordering']
        category_stats = analysis['category_stats']
        total_moves = analysis['total_moves']
        
        # Summary statistics
        print(f"\n📊 MOVE BREAKDOWN:")
        print(f"  Total moves: {total_moves}")
        for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_moves) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        # Top 10 optimal moves
        print(f"\n🎯 TOP 10 V13.x OPTIMAL ORDERING:")
        print("-" * 50)
        for i, analysis in enumerate(optimal_ordering[:10], 1):
            categories_str = ", ".join(analysis['categories'][:2])
            print(f"  {i:2d}. {analysis['description']:15s} ({analysis['priority_score']:4d}) - {categories_str}")
        
        # Top 10 engine moves
        print(f"\n⚙️  TOP 10 ENGINE ORDERING:")
        print("-" * 50)
        # Store the original board for engine move analysis
        original_board = analysis.get('board')  
        for i, move in enumerate(engine_ordering[:10], 1):
            if original_board:
                categories, score = self.simple_categorize_move(original_board, move)
                categories_str = ", ".join(categories[:2])
                description = f"{move.uci()}"
                print(f"  {i:2d}. {description:15s} ({score:4d}) - {categories_str}")
            else:
                print(f"  {i:2d}. {move.uci():15s}")  # Fallback if no board available
        
        # Comparison analysis
        print(f"\n⚔️  ORDERING COMPARISON (Top 10):")
        print("-" * 50)
        
        optimal_top10 = [a['move'] for a in optimal_ordering[:10]]
        engine_top10 = engine_ordering[:10]
        
        matches = sum(1 for move in optimal_top10 if move in engine_top10)
        match_percentage = (matches / 10) * 100
        
        print(f"  Matches in top 10: {matches}/10 ({match_percentage:.0f}%)")
        
        # Critical issues
        optimal_critical = [a for a in optimal_ordering[:5] if a['priority_score'] > 400]
        engine_critical = [move for move in engine_ordering[:5]]
        
        missed_critical = [a for a in optimal_critical if a['move'] not in engine_critical]
        
        if missed_critical:
            print(f"\n🚨 CRITICAL MOVES MISSED BY ENGINE:")
            for analysis in missed_critical:
                print(f"    ❌ {analysis['description']} ({analysis['priority_score']}) - {', '.join(analysis['categories'])}")
        else:
            print(f"\n✅ Engine correctly prioritized critical moves")
        
        # V13.x Recommendations
        print(f"\n💡 V13.x RECOMMENDATIONS:")
        print("-" * 50)
        
        recommendations = []
        
        if match_percentage < 70:
            recommendations.append(f"🎯 Improve move ordering - only {match_percentage:.0f}% agreement with optimal")
        
        tactical_count = sum(1 for a in move_analyses if a['is_tactical'])
        tactical_in_top5 = sum(1 for a in optimal_ordering[:5] if a['is_tactical'])
        if tactical_count > 0 and tactical_in_top5 > 0:
            engine_tactical_top5 = sum(1 for move in engine_ordering[:5] if move in [a['move'] for a in optimal_ordering[:5] if a['is_tactical']])
            if engine_tactical_top5 < tactical_in_top5:
                recommendations.append(f"⚡ Prioritize {tactical_in_top5 - engine_tactical_top5} tactical moves higher")
        
        captures = category_stats.get("Good Capture", 0)
        if captures > 0:
            recommendations.append(f"🎯 Ensure {captures} good captures are prioritized")
        
        if not recommendations:
            recommendations.append("✅ Move ordering appears well-optimized")
        
        for rec in recommendations:
            print(f"  {rec}")

def test_simple_v13():
    """Test simple V13.x analyzer on key positions"""
    analyzer = SimpleV13MoveAnalyzer()
    
    test_positions = {
        "Opening": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Middlegame": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
        "Tactical": "r2qkb1r/pp2nppp/3p1n2/2pP4/2P1P3/2N2N2/PP3PPP/R1BQKB1R w KQq - 0 6",
        "Complex": "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NP1/PPP1NPB1/R1BQ1RK1 b - - 0 9"
    }
    
    print("🚀 V7P3R Simple V13.x Move Ordering Analysis")
    print("Testing on weakness-prone positions")
    print("="*60)
    
    for position_name, fen in test_positions.items():
        board = chess.Board(fen)
        analysis = analyzer.analyze_simple_ordering(board)
        analysis['fen'] = fen
        analyzer.print_simple_analysis(analysis, position_name)
        print("\n" + "="*30)

if __name__ == "__main__":
    test_simple_v13()