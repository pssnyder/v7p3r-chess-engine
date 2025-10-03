#!/usr/bin/env python3
"""
V7P3R Heuristic Analysis Utility
================================
Analyzes the balance and impact of evaluation heuristics to identify
"hot spots" that may be drawing focus away from optimal moves.

Features:
1. Core Evaluation Breakdown - scores across all eval functions
2. Move Ordering Impact Analysis - ratio analysis of move ordering
3. Position-specific Heuristic Firing - which heuristics activate
4. Score Balance Analysis - relative weights and impacts

Author: Pat Snyder for V7P3R v12.4 Enhanced Castling Analysis
"""

import sys
import os
import chess
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.v7p3r import V7P3REngine
from src.v7p3r_bitboard_evaluator import V7P3RBitboardEvaluator

@dataclass
class HeuristicBreakdown:
    """Detailed breakdown of heuristic scores for a position"""
    position_fen: str
    total_score: float
    
    # Core Evaluation Components
    material_score: float = 0.0
    piece_square_score: float = 0.0
    mobility_score: float = 0.0
    king_safety_score: float = 0.0
    castling_score: float = 0.0
    pawn_structure_score: float = 0.0
    piece_coordination_score: float = 0.0
    
    # Move Ordering Components
    hash_move_bonus: float = 0.0
    killer_move_bonus: float = 0.0
    history_bonus: float = 0.0
    capture_score: float = 0.0
    promotion_bonus: float = 0.0
    
    # Tactical Components
    tactical_bonus: float = 0.0
    threat_evaluation: float = 0.0
    piece_activity: float = 0.0
    
    # Meta-analysis
    heuristics_firing: List[str] = field(default_factory=list)
    score_ratios: Dict[str, float] = field(default_factory=dict)
    hotspots: List[str] = field(default_factory=list)

class V7P3RHeuristicAnalyzer:
    """Comprehensive heuristic analysis for V7P3R engine"""
    
    def __init__(self):
        self.engine = V7P3REngine()
        self.bitboard_evaluator = self.engine.bitboard_evaluator.bitboard_evaluator
    
    def analyze_position(self, board: chess.Board, color: chess.Color = chess.WHITE) -> HeuristicBreakdown:
        """Perform comprehensive heuristic analysis of a position"""
        
        breakdown = HeuristicBreakdown(
            position_fen=board.fen(),
            total_score=0.0
        )
        
        print(f"\n{'='*60}")
        print(f"V7P3R HEURISTIC ANALYSIS")
        print(f"{'='*60}")
        print(f"Position: {board.fen()}")
        print(f"Analyzing for: {'WHITE' if color == chess.WHITE else 'BLACK'}")
        print(f"Move: {len(board.move_stack) + 1}")
        print()
        print("Board:")
        print(board)
        print()
        
        # Core Evaluation Analysis
        self._analyze_core_evaluation(board, color, breakdown)
        
        # Move Ordering Analysis
        self._analyze_move_ordering(board, color, breakdown)
        
        # Tactical Analysis
        self._analyze_tactical_components(board, color, breakdown)
        
        # Meta-analysis
        self._perform_meta_analysis(breakdown)
        
        return breakdown
    
    def _analyze_core_evaluation(self, board: chess.Board, color: chess.Color, breakdown: HeuristicBreakdown):
        """Analyze core evaluation components"""
        
        print("🔍 CORE EVALUATION ANALYSIS")
        print("-" * 40)
        
        # Get bitboard data
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        white_knights = board.pieces(chess.KNIGHT, chess.WHITE)
        black_knights = board.pieces(chess.KNIGHT, chess.BLACK)
        white_bishops = board.pieces(chess.BISHOP, chess.WHITE)
        black_bishops = board.pieces(chess.BISHOP, chess.BLACK)
        white_rooks = board.pieces(chess.ROOK, chess.WHITE)
        black_rooks = board.pieces(chess.ROOK, chess.BLACK)
        white_queens = board.pieces(chess.QUEEN, chess.WHITE)
        black_queens = board.pieces(chess.QUEEN, chess.BLACK)
        white_king = board.pieces(chess.KING, chess.WHITE)
        black_king = board.pieces(chess.KING, chess.BLACK)
        
        # Material Analysis
        material_white = (len(white_pawns) * 100 + len(white_knights) * 320 + 
                         len(white_bishops) * 330 + len(white_rooks) * 500 + 
                         len(white_queens) * 900)
        material_black = (len(black_pawns) * 100 + len(black_knights) * 320 + 
                         len(black_bishops) * 330 + len(black_rooks) * 500 + 
                         len(black_queens) * 900)
        
        breakdown.material_score = material_white - material_black
        if color == chess.BLACK:
            breakdown.material_score = -breakdown.material_score
        
        print(f"📊 Material Balance: {breakdown.material_score:+.1f}")
        print(f"   White: {material_white} | Black: {material_black}")
        
        # Piece Square Table Analysis  
        pst_score = self._calculate_piece_square_score(board)
        breakdown.piece_square_score = pst_score if color == chess.WHITE else -pst_score
        print(f"🎯 Piece Square Tables: {breakdown.piece_square_score:+.1f}")
        
        # Mobility Analysis
        mobility_score = self._calculate_mobility_score(board, color)
        breakdown.mobility_score = mobility_score
        print(f"🏃 Mobility Score: {breakdown.mobility_score:+.1f}")
        
        # King Safety & Castling Analysis
        castling_score = self.bitboard_evaluator._evaluate_enhanced_castling(board, color)
        breakdown.castling_score = castling_score
        king_safety = self._analyze_king_safety(board, color)
        breakdown.king_safety_score = king_safety
        print(f"🏰 Castling Evaluation: {breakdown.castling_score:+.1f}")
        print(f"👑 King Safety: {breakdown.king_safety_score:+.1f}")
        
        # Pawn Structure Analysis
        pawn_score = self._analyze_pawn_structure(board, color)
        breakdown.pawn_structure_score = pawn_score
        print(f"♟️  Pawn Structure: {breakdown.pawn_structure_score:+.1f}")
        
        # Total core score
        breakdown.total_score = (breakdown.material_score + breakdown.piece_square_score + 
                               breakdown.mobility_score + breakdown.castling_score + 
                               breakdown.king_safety_score + breakdown.pawn_structure_score)
        
        print(f"📈 Core Total: {breakdown.total_score:+.1f}")
        print()
    
    def _analyze_move_ordering(self, board: chess.Board, color: chess.Color, breakdown: HeuristicBreakdown):
        """Analyze move ordering heuristics"""
        
        print("🎯 MOVE ORDERING ANALYSIS")
        print("-" * 40)
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            print("No legal moves available")
            return
        
        move_scores = []
        
        for move in legal_moves[:10]:  # Analyze top 10 moves
            score = self._evaluate_move_ordering_score(board, move)
            move_scores.append((move, score))
        
        # Sort by score
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("Top Move Ordering Scores:")
        for i, (move, score) in enumerate(move_scores[:5]):
            move_type = self._classify_move_type(board, move)
            print(f"  {i+1}. {move} ({move_type}): {score:+.1f}")
        
        # Analyze ordering components
        if move_scores:
            best_move = move_scores[0][0]
            breakdown.capture_score = self._analyze_capture_value(board, best_move)
            breakdown.history_bonus = self._get_history_bonus(best_move)
            
            print(f"\nBest Move Analysis ({best_move}):")
            print(f"  🎯 Capture Value: {breakdown.capture_score:+.1f}")
            print(f"  📚 History Bonus: {breakdown.history_bonus:+.1f}")
        
        print()
    
    def _analyze_tactical_components(self, board: chess.Board, color: chess.Color, breakdown: HeuristicBreakdown):
        """Analyze tactical evaluation components"""
        
        print("⚔️  TACTICAL ANALYSIS")
        print("-" * 40)
        
        # Piece Activity
        activity_score = self._calculate_piece_activity(board, color)
        breakdown.piece_activity = activity_score
        print(f"🎮 Piece Activity: {breakdown.piece_activity:+.1f}")
        
        # Threat Evaluation
        threat_score = self._evaluate_threats(board, color)
        breakdown.threat_evaluation = threat_score
        print(f"⚡ Threat Evaluation: {breakdown.threat_evaluation:+.1f}")
        
        # Tactical Patterns
        tactical_bonus = self._detect_tactical_patterns(board, color)
        breakdown.tactical_bonus = tactical_bonus
        print(f"🧩 Tactical Patterns: {breakdown.tactical_bonus:+.1f}")
        
        print()
    
    def _perform_meta_analysis(self, breakdown: HeuristicBreakdown):
        """Perform meta-analysis to identify hotspots and balance issues"""
        
        print("🔬 META-ANALYSIS")
        print("-" * 40)
        
        # Calculate score components
        components = {
            'Material': abs(breakdown.material_score),
            'Piece Squares': abs(breakdown.piece_square_score),
            'Mobility': abs(breakdown.mobility_score),
            'Castling': abs(breakdown.castling_score),
            'King Safety': abs(breakdown.king_safety_score),
            'Pawn Structure': abs(breakdown.pawn_structure_score),
            'Piece Activity': abs(breakdown.piece_activity),
            'Threats': abs(breakdown.threat_evaluation),
            'Tactical': abs(breakdown.tactical_bonus)
        }
        
        total_abs_score = sum(components.values())
        
        if total_abs_score > 0:
            # Calculate ratios
            for component, score in components.items():
                ratio = (score / total_abs_score) * 100
                breakdown.score_ratios[component] = ratio
                
                # Identify hotspots (>25% of total evaluation)
                if ratio > 25:
                    breakdown.hotspots.append(f"{component} ({ratio:.1f}%)")
            
            # Identify firing heuristics
            for component, score in components.items():
                if score > 5:  # Threshold for "firing"
                    breakdown.heuristics_firing.append(component)
        
        print("Score Distribution:")
        for component, ratio in sorted(breakdown.score_ratios.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(ratio / 5)  # Visual bar
            print(f"  {component:15}: {ratio:5.1f}% {bar}")
        
        if breakdown.hotspots:
            print(f"\n🔥 HOTSPOTS DETECTED:")
            for hotspot in breakdown.hotspots:
                print(f"  ⚠️  {hotspot}")
        
        if breakdown.heuristics_firing:
            print(f"\n✅ ACTIVE HEURISTICS:")
            for heuristic in breakdown.heuristics_firing:
                print(f"  🎯 {heuristic}")
        
        print()
    
    # Helper methods for detailed analysis
    
    def _calculate_piece_square_score(self, board: chess.Board) -> float:
        """Calculate piece square table contribution"""
        # Simplified PST calculation
        score = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Basic center control bonus
                center_bonus = 0
                if square in [chess.D4, chess.D5, chess.E4, chess.E5]:
                    center_bonus = 20
                elif square in [chess.C3, chess.C4, chess.C5, chess.C6,
                               chess.D3, chess.D6, chess.E3, chess.E6,
                               chess.F3, chess.F4, chess.F5, chess.F6]:
                    center_bonus = 10
                
                if piece.color == chess.WHITE:
                    score += center_bonus
                else:
                    score -= center_bonus
        
        return score
    
    def _calculate_mobility_score(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate mobility score for the given color"""
        mobility = len(list(board.legal_moves))
        
        # Test opponent mobility
        board.push(chess.Move.null())
        opponent_mobility = len(list(board.legal_moves))
        board.pop()
        
        mobility_diff = mobility - opponent_mobility
        return mobility_diff * 2.0  # Weight mobility
    
    def _analyze_king_safety(self, board: chess.Board, color: chess.Color) -> float:
        """Analyze king safety beyond castling"""
        king_square = board.king(color)
        if king_square is None:
            return 0.0
            
        safety_score = 0.0
        
        # Check for attacks on king
        attackers = board.attackers(not color, king_square)
        safety_score -= len(attackers) * 15
        
        # Check for pawn shield (basic)
        if color == chess.WHITE:
            shield_squares = [king_square + 8, king_square + 7, king_square + 9]
        else:
            shield_squares = [king_square - 8, king_square - 7, king_square - 9]
        
        for square in shield_squares:
            if 0 <= square < 64:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    safety_score += 10
        
        return safety_score
    
    def _analyze_pawn_structure(self, board: chess.Board, color: chess.Color) -> float:
        """Analyze pawn structure contribution"""
        pawns = board.pieces(chess.PAWN, color)
        opponent_pawns = board.pieces(chess.PAWN, not color)
        
        structure_score = 0.0
        
        # Passed pawns
        for pawn_square in pawns:
            file = chess.square_file(pawn_square)
            rank = chess.square_rank(pawn_square)
            
            is_passed = True
            if color == chess.WHITE:
                for opp_square in opponent_pawns:
                    opp_file = chess.square_file(opp_square)
                    opp_rank = chess.square_rank(opp_square)
                    if abs(opp_file - file) <= 1 and opp_rank > rank:
                        is_passed = False
                        break
            else:
                for opp_square in opponent_pawns:
                    opp_file = chess.square_file(opp_square)
                    opp_rank = chess.square_rank(opp_square)
                    if abs(opp_file - file) <= 1 and opp_rank < rank:
                        is_passed = False
                        break
            
            if is_passed:
                structure_score += 25
        
        return structure_score
    
    def _evaluate_move_ordering_score(self, board: chess.Board, move: chess.Move) -> float:
        """Evaluate move ordering score for a move"""
        score = 0.0
        
        # Capture value
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                score += self._get_piece_value(captured_piece.piece_type) / 10
        
        # Promotion bonus
        if move.promotion:
            score += self._get_piece_value(move.promotion) / 10
        
        # Center control
        if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
            score += 20
        
        # Development bonus
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            back_rank = 0 if piece.color == chess.WHITE else 7  # RANK_1 = 0, RANK_8 = 7
            if chess.square_rank(move.from_square) == back_rank:
                score += 10
        
        return score
    
    def _classify_move_type(self, board: chess.Board, move: chess.Move) -> str:
        """Classify the type of move"""
        if board.is_castling(move):
            return "Castling"
        elif board.is_capture(move):
            return "Capture"
        elif move.promotion:
            return "Promotion"
        elif board.gives_check(move):  # Use gives_check instead of is_check
            return "Check"
        else:
            piece = board.piece_at(move.from_square)
            if piece:
                piece_names = {
                    chess.PAWN: "Pawn",
                    chess.KNIGHT: "Knight", 
                    chess.BISHOP: "Bishop",
                    chess.ROOK: "Rook",
                    chess.QUEEN: "Queen",
                    chess.KING: "King"
                }
                return f"{piece_names.get(piece.piece_type, 'Unknown')} Move"
            return "Unknown"
    
    def _analyze_capture_value(self, board: chess.Board, move: chess.Move) -> float:
        """Analyze capture value of a move"""
        if not board.is_capture(move):
            return 0.0
        
        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            return self._get_piece_value(captured_piece.piece_type)
        return 0.0
    
    def _get_history_bonus(self, move: chess.Move) -> float:
        """Get history bonus for a move (simplified)"""
        # Simplified history bonus
        return 5.0 if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5] else 0.0
    
    def _calculate_piece_activity(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate piece activity score"""
        activity = 0.0
        
        pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        for piece_type in pieces:
            for square in board.pieces(piece_type, color):
                attacks = len(board.attacks(square))
                activity += attacks * 2
        
        return activity
    
    def _evaluate_threats(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate threats in the position"""
        threat_score = 0.0
        
        # Look for hanging pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != color:
                attackers = board.attackers(color, square)
                defenders = board.attackers(piece.color, square)
                
                if len(attackers) > len(defenders):
                    threat_score += self._get_piece_value(piece.piece_type) / 4
        
        return threat_score
    
    def _detect_tactical_patterns(self, board: chess.Board, color: chess.Color) -> float:
        """Detect tactical patterns"""
        tactical_score = 0.0
        
        # Simple tactical pattern detection
        # Look for discovered attacks, pins, etc.
        # This is a simplified version
        
        legal_moves = list(board.legal_moves)
        for move in legal_moves[:5]:  # Check first 5 moves
            board.push(move)
            if board.is_check():
                tactical_score += 10
            board.pop()
        
        return tactical_score
    
    def _get_piece_value(self, piece_type: chess.PieceType) -> float:
        """Get standard piece values"""
        values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        return values.get(piece_type, 0)

def analyze_test_positions():
    """Analyze a set of test positions to identify evaluation patterns"""
    
    analyzer = V7P3RHeuristicAnalyzer()
    
    test_positions = [
        {
            'name': 'Starting Position',
            'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'description': 'Standard opening position'
        },
        {
            'name': 'Castling Critical Position',
            'fen': 'r1bqk1nr/p2ppppp/1p6/8/8/8/PPPP1PPP/RNBQK1NR w KQkq - 0 6',
            'description': 'Position where v12.2 chose Kf1'
        },
        {
            'name': 'Middle Game Position',
            'fen': 'r1bq1rk1/pp2nppp/2n1p3/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 8',
            'description': 'Typical middlegame position'
        },
        {
            'name': 'Tactical Position',
            'fen': 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4',
            'description': 'Position with tactical possibilities'
        }
    ]
    
    print("🔬 V7P3R HEURISTIC ANALYSIS SUITE")
    print("=" * 80)
    print("Analyzing evaluation balance and hotspots across multiple positions")
    print()
    
    all_breakdowns = []
    
    for i, position in enumerate(test_positions, 1):
        print(f"\n🎯 POSITION {i}: {position['name']}")
        print(f"📝 {position['description']}")
        
        board = chess.Board(position['fen'])
        breakdown = analyzer.analyze_position(board, chess.WHITE)
        all_breakdowns.append((position['name'], breakdown))
    
    # Summary analysis
    print("\n" + "="*80)
    print("📊 SUMMARY ANALYSIS")
    print("="*80)
    
    # Common hotspots
    all_hotspots = []
    for name, breakdown in all_breakdowns:
        all_hotspots.extend(breakdown.hotspots)
    
    if all_hotspots:
        print("\n🔥 FREQUENT HOTSPOTS:")
        from collections import Counter
        hotspot_counts = Counter(all_hotspots)
        for hotspot, count in hotspot_counts.most_common():
            print(f"  {hotspot} (appears in {count} positions)")
    
    # Common active heuristics
    all_heuristics = []
    for name, breakdown in all_breakdowns:
        all_heuristics.extend(breakdown.heuristics_firing)
    
    if all_heuristics:
        print("\n✅ MOST ACTIVE HEURISTICS:")
        from collections import Counter
        heuristic_counts = Counter(all_heuristics)
        for heuristic, count in heuristic_counts.most_common():
            print(f"  {heuristic} (active in {count} positions)")
    
    print("\n📈 RECOMMENDATIONS:")
    print("  1. Monitor hotspots for potential over-weighting")
    print("  2. Ensure tactical coverage is appropriate")
    print("  3. Balance positional vs tactical evaluation")
    print("  4. Consider adjusting weights for dominant heuristics")

if __name__ == "__main__":
    analyze_test_positions()