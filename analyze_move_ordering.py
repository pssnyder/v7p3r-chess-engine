#!/usr/bin/env python3
"""
V7P3R Move Ordering Analyzer
Analyzes move ordering performance and categorization for optimization insights
"""

import chess
import sys
import os
import time
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from v7p3r import V7P3REngine
except ImportError:
    print("Warning: Could not import V7P3REngine, using fallback analysis")
    V7P3REngine = None


@dataclass
class MoveCategory:
    """Represents a category of moves with analysis"""
    name: str
    moves: List[chess.Move] = field(default_factory=list)
    descriptions: List[str] = field(default_factory=list)
    scores: List[int] = field(default_factory=list)
    
    def add_move(self, move: chess.Move, description: str = "", score: int = 0):
        self.moves.append(move)
        self.descriptions.append(description)
        self.scores.append(score)
    
    def __len__(self):
        return len(self.moves)


class V7P3RMoveAnalyzer:
    """Comprehensive move ordering analysis system"""
    
    def __init__(self):
        self.engine = V7P3REngine() if V7P3REngine else None
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320, 
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
    def categorize_move(self, board: chess.Board, move: chess.Move) -> List[str]:
        """Categorize a move into multiple categories"""
        categories = []
        
        # Basic move properties
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                victim_value = self.piece_values.get(victim.piece_type, 0)
                attacker_value = self.piece_values.get(attacker.piece_type, 0)
                if victim_value >= attacker_value:
                    categories.append("Good Capture")
                else:
                    categories.append("Bad Capture")
                
                # MVV-LVA analysis
                categories.append(f"Capture: {chess.piece_name(victim.piece_type)} x {chess.piece_name(attacker.piece_type)}")
        
        # Check if move gives check
        board.push(move)
        if board.is_check():
            categories.append("Check")
            
            # Check if it's checkmate
            if board.is_checkmate():
                categories.append("Checkmate")
                
        # Check if it's stalemate
        if board.is_stalemate():
            categories.append("Stalemate")
            
        board.pop()
        
        # Promotion
        if move.promotion:
            categories.append(f"Promotion to {chess.piece_name(move.promotion)}")
            if move.promotion == chess.QUEEN:
                categories.append("Queen Promotion")
        
        # Castling
        if board.is_castling(move):
            if move.to_square > move.from_square:
                categories.append("Kingside Castle")
            else:
                categories.append("Queenside Castle")
        
        # En passant
        if board.is_en_passant(move):
            categories.append("En Passant")
            
        # Piece-specific analysis
        piece = board.piece_at(move.from_square)
        if piece:
            # Pawn moves
            if piece.piece_type == chess.PAWN:
                # Pawn push analysis
                rank_diff = abs(chess.square_rank(move.to_square) - chess.square_rank(move.from_square))
                if rank_diff == 2:
                    categories.append("Pawn Double Push")
                elif rank_diff == 1:
                    categories.append("Pawn Push")
                    
            # King moves in endgame
            elif piece.piece_type == chess.KING:
                piece_count = len(board.piece_map())
                if piece_count <= 10:  # Endgame
                    categories.append("King Activity (Endgame)")
                    
            # Knight moves
            elif piece.piece_type == chess.KNIGHT:
                # Check for knight forks
                board.push(move)
                attacks = 0
                high_value_attacks = 0
                for sq in chess.SQUARES:
                    if board.is_attacked_by(piece.color, sq):
                        target = board.piece_at(sq)
                        if target and target.color != piece.color:
                            attacks += 1
                            if target.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                                high_value_attacks += 1
                if attacks >= 2:
                    categories.append("Knight Fork")
                    if high_value_attacks >= 1:
                        categories.append("Knight Fork (High Value)")
                board.pop()
                
            # Piece development
            if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                start_rank = chess.square_rank(move.from_square)
                if piece.color == chess.WHITE and start_rank == 0:
                    categories.append("Piece Development")
                elif piece.color == chess.BLACK and start_rank == 7:
                    categories.append("Piece Development")
        
        # If no categories found, it's a quiet move
        if not categories:
            categories.append("Quiet Move")
            
        return categories
    
    def analyze_position_moves(self, board: chess.Board, max_moves: int = None) -> Dict[str, MoveCategory]:
        """Analyze all legal moves in a position"""
        legal_moves = list(board.legal_moves)
        if max_moves:
            legal_moves = legal_moves[:max_moves]
            
        move_categories = defaultdict(lambda: MoveCategory(""))
        
        print(f"📋 Analyzing {len(legal_moves)} legal moves...")
        
        for move in legal_moves:
            categories = self.categorize_move(board, move)
            
            for category in categories:
                if not move_categories[category].name:
                    move_categories[category].name = category
                    
                move_description = f"{move} ({chess.square_name(move.from_square)}-{chess.square_name(move.to_square)})"
                move_categories[category].add_move(move, move_description)
        
        return dict(move_categories)
    
    def get_engine_move_ordering(self, board: chess.Board) -> Tuple[List[chess.Move], Dict]:
        """Get move ordering from the engine if available"""
        if not self.engine:
            return list(board.legal_moves), {}
            
        # Try to access the engine's move ordering
        try:
            # Check if engine has move ordering methods
            if hasattr(self.engine, '_order_moves'):
                ordered_moves = self.engine._order_moves(board, list(board.legal_moves))
                return ordered_moves, {}
            elif hasattr(self.engine, 'order_moves'):
                ordered_moves = self.engine.order_moves(board)
                return ordered_moves, {}
            else:
                # Fallback: use simple MVV-LVA ordering
                return self._simple_move_ordering(board), {}
        except Exception as e:
            print(f"Warning: Could not get engine move ordering: {e}")
            return self._simple_move_ordering(board), {}
    
    def _simple_move_ordering(self, board: chess.Board) -> List[chess.Move]:
        """Simple fallback move ordering"""
        moves = list(board.legal_moves)
        
        def move_score(move):
            score = 0
            
            # Captures (MVV-LVA)
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    victim_value = self.piece_values.get(victim.piece_type, 0)
                    attacker_value = self.piece_values.get(attacker.piece_type, 0)
                    score += (victim_value - attacker_value // 10) * 100
            
            # Checks
            board.push(move)
            if board.is_check():
                score += 500
                if board.is_checkmate():
                    score += 10000
            board.pop()
            
            # Promotions
            if move.promotion == chess.QUEEN:
                score += 800
            
            return score
        
        moves.sort(key=move_score, reverse=True)
        return moves
    
    def compare_move_selection(self, board: chess.Board, top_n: int = 10) -> Dict:
        """Compare all moves vs engine's top selections"""
        # Get all moves categorized
        all_moves_categorized = self.analyze_position_moves(board)
        
        # Get engine's move ordering
        engine_ordered_moves, engine_stats = self.get_engine_move_ordering(board)
        
        # Take top N moves from engine
        selected_moves = engine_ordered_moves[:top_n]
        discarded_moves = engine_ordered_moves[top_n:]
        
        # Categorize selected and discarded moves
        selected_categorized = defaultdict(lambda: MoveCategory(""))
        discarded_categorized = defaultdict(lambda: MoveCategory(""))
        
        for move in selected_moves:
            categories = self.categorize_move(board, move)
            for category in categories:
                if not selected_categorized[category].name:
                    selected_categorized[category].name = category
                move_desc = f"{move} ({chess.square_name(move.from_square)}-{chess.square_name(move.to_square)})"
                selected_categorized[category].add_move(move, move_desc)
        
        for move in discarded_moves:
            categories = self.categorize_move(board, move)
            for category in categories:
                if not discarded_categorized[category].name:
                    discarded_categorized[category].name = category
                move_desc = f"{move} ({chess.square_name(move.from_square)}-{chess.square_name(move.to_square)})"
                discarded_categorized[category].add_move(move, move_desc)
        
        return {
            'all_moves': all_moves_categorized,
            'selected_moves': dict(selected_categorized),
            'discarded_moves': dict(discarded_categorized),
            'engine_order': engine_ordered_moves,
            'engine_stats': engine_stats,
            'top_n': top_n
        }
    
    def print_move_analysis(self, analysis: Dict, position_name: str = "Position"):
        """Print comprehensive move analysis"""
        print(f"\n{'='*80}")
        print(f"🔍 MOVE ORDERING ANALYSIS: {position_name}")
        print(f"{'='*80}")
        
        all_moves = analysis['all_moves']
        selected_moves = analysis['selected_moves']
        discarded_moves = analysis['discarded_moves']
        top_n = analysis['top_n']
        
        # Summary statistics
        total_moves = sum(len(cat.moves) for cat in all_moves.values())
        selected_count = sum(len(cat.moves) for cat in selected_moves.values())
        discarded_count = sum(len(cat.moves) for cat in discarded_moves.values())
        
        print(f"\n📊 SUMMARY:")
        print(f"  Total legal moves: {total_moves}")
        print(f"  Top {top_n} selected: {selected_count}")
        print(f"  Discarded: {discarded_count}")
        
        # All moves breakdown
        print(f"\n📋 ALL AVAILABLE MOVES BY CATEGORY:")
        print("-" * 60)
        for category_name, category in sorted(all_moves.items(), key=lambda x: len(x[1].moves), reverse=True):
            print(f"  {category_name}: {len(category)} moves")
            for i, (move, desc) in enumerate(zip(category.moves[:3], category.descriptions[:3])):
                print(f"    {i+1}. {desc}")
            if len(category) > 3:
                print(f"    ... and {len(category) - 3} more")
        
        # Selected moves analysis
        print(f"\n✅ TOP {top_n} MOVES SELECTED BY ENGINE:")
        print("-" * 60)
        if selected_moves:
            for category_name, category in sorted(selected_moves.items(), key=lambda x: len(x[1].moves), reverse=True):
                print(f"  {category_name}: {len(category)} moves")
                for move, desc in zip(category.moves, category.descriptions):
                    print(f"    • {desc}")
        else:
            print("  No categorized moves in top selection")
        
        # Discarded moves analysis
        print(f"\n❌ DISCARDED MOVES BY ENGINE:")
        print("-" * 60)
        if discarded_moves:
            for category_name, category in sorted(discarded_moves.items(), key=lambda x: len(x[1].moves), reverse=True):
                print(f"  {category_name}: {len(category)} moves")
                for i, (move, desc) in enumerate(zip(category.moves[:2], category.descriptions[:2])):
                    print(f"    • {desc}")
                if len(category) > 2:
                    print(f"    ... and {len(category) - 2} more discarded")
        else:
            print("  All moves were selected")
        
        # Move ordering quality analysis
        print(f"\n🎯 MOVE ORDERING QUALITY ANALYSIS:")
        print("-" * 60)
        
        # Check for potential issues
        issues = []
        
        # Check if good captures were discarded
        if 'Good Capture' in discarded_moves:
            good_captures_discarded = len(discarded_moves['Good Capture'])
            issues.append(f"⚠️  {good_captures_discarded} good captures were discarded")
        
        # Check if checks were prioritized
        checks_selected = len(selected_moves.get('Check', MoveCategory('')))
        checks_total = len(all_moves.get('Check', MoveCategory('')))
        if checks_total > 0:
            check_priority = (checks_selected / checks_total) * 100
            if check_priority < 50:
                issues.append(f"⚠️  Only {check_priority:.1f}% of checks were prioritized")
            else:
                print(f"✅ Good check prioritization: {check_priority:.1f}%")
        
        # Check if bad captures were deprioritized  
        bad_captures_selected = len(selected_moves.get('Bad Capture', MoveCategory('')))
        bad_captures_total = len(all_moves.get('Bad Capture', MoveCategory('')))
        if bad_captures_total > 0 and bad_captures_selected > 0:
            bad_capture_rate = (bad_captures_selected / bad_captures_total) * 100
            if bad_capture_rate > 30:
                issues.append(f"⚠️  {bad_capture_rate:.1f}% of bad captures were prioritized")
        
        # Print issues or success
        if issues:
            for issue in issues:
                print(f"  {issue}")
        else:
            print("  ✅ No obvious move ordering issues detected")
    
    def test_multiple_positions(self):
        """Test move ordering on various chess positions"""
        test_positions = {
            "Starting Position": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "Middle Game": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
            "Tactical Position": "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
            "Endgame": "8/2k5/3p4/p2P1p2/P2P1P2/8/2K5/8 w - - 0 1",
            "Complex Middlegame": "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NP1/PPP1NPB1/R1BQ1RK1 b - - 0 9"
        }
        
        for position_name, fen in test_positions.items():
            board = chess.Board(fen)
            analysis = self.compare_move_selection(board, top_n=8)
            self.print_move_analysis(analysis, position_name)
            
            print(f"\n🔗 Engine Move Order (Top 8):")
            for i, move in enumerate(analysis['engine_order'][:8], 1):
                categories = self.categorize_move(board, move)
                cat_str = ", ".join(categories[:2])  # Show first 2 categories
                print(f"  {i}. {move} ({chess.square_name(move.from_square)}-{chess.square_name(move.to_square)}) - {cat_str}")
        
        return test_positions


def main():
    """Run comprehensive move ordering analysis"""
    print("🚀 V7P3R Move Ordering Analysis System")
    print("=" * 60)
    
    analyzer = V7P3RMoveAnalyzer()
    
    # Test on multiple positions
    test_positions = analyzer.test_multiple_positions()
    
    print(f"\n📈 OVERALL ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"✅ Tested {len(test_positions)} different position types")
    print("✅ Analyzed move categorization and engine prioritization")
    print("✅ Identified potential move ordering improvements")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("1. Review any good captures that were discarded")
    print("2. Ensure checks are properly prioritized") 
    print("3. Verify bad captures are deprioritized")
    print("4. Consider tactical patterns in move ordering")
    print("5. Test move ordering improvements iteratively")


if __name__ == "__main__":
    main()