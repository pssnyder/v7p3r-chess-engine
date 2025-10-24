#!/usr/bin/env python3
"""
Enhanced V7P3R Move Ordering Analyzer v13.x
Analyzes and improves move ordering based on V12.6 weakness analysis findings

Key Improvements Based on Analysis:
- 75% of bad moves weren't in top 5 -> Fix move ordering priority
- 70% tactical misses -> Enhance tactical move detection
- 27.7% hanging piece misses -> Add piece safety priority
- Time pressure errors -> Optimize for faster search

This analyzer focuses on:
1. Legal moves categorization 
2. Engine move ordering vs optimal ordering
3. Move volume tuning for search speed
4. Tactical move prioritization fixes
"""

import chess
import sys
import os
import time
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from v7p3r import V7P3REngine
except ImportError:
    print("Warning: Could not import V7P3REngine, using enhanced fallback analysis")
    V7P3REngine = None


@dataclass
class MoveOrderingStats:
    """Statistics for move ordering analysis"""
    total_legal_moves: int = 0
    tactical_moves: int = 0
    hanging_piece_saves: int = 0
    captures: int = 0
    checks: int = 0
    quiet_moves: int = 0
    
    # V13.x specific metrics
    high_priority_moves: int = 0  # Moves that should be searched first
    medium_priority_moves: int = 0
    low_priority_moves: int = 0
    
    # Performance metrics
    search_reduction_potential: float = 0.0  # % of moves we can skip


@dataclass
class EnhancedMoveCategory:
    """Enhanced move category with V13.x improvements"""
    name: str
    priority_level: int  # 1=highest, 5=lowest
    moves: List[chess.Move] = field(default_factory=list)
    descriptions: List[str] = field(default_factory=list)
    scores: List[int] = field(default_factory=list)
    tactical_values: List[int] = field(default_factory=list)  # Tactical importance
    
    def add_move(self, move: chess.Move, description: str = "", score: int = 0, tactical_value: int = 0):
        self.moves.append(move)
        self.descriptions.append(description)
        self.scores.append(score)
        self.tactical_values.append(tactical_value)
    
    def __len__(self):
        return len(self.moves)


class EnhancedV7P3RMoveAnalyzer:
    """Enhanced move ordering analysis system for V13.x"""
    
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
        
        # V13.x Enhancement: Move priority system based on weakness analysis
        self.move_priorities = {
            # Priority 1: Critical tactical moves (missed 70% in V12.6)
            "Checkmate": 1,
            "Good Capture": 1,
            "Piece Safety (Save Hanging)": 1,
            "Tactical Fork": 1,
            "Tactical Pin": 1,
            "Tactical Skewer": 1,
            
            # Priority 2: Important moves
            "Check": 2,
            "Queen Promotion": 2,
            "Piece Development": 2,
            "Castle": 2,
            
            # Priority 3: Standard moves
            "Equal Capture": 3,
            "Pawn Push": 3,
            "Knight Fork": 3,
            
            # Priority 4: Dubious moves
            "Bad Capture": 4,
            "Quiet Move": 4,
            
            # Priority 5: Usually bad moves
            "Stalemate": 5,
            "King Exposure": 5
        }
    
    def detect_hanging_pieces(self, board: chess.Board) -> Set[chess.Square]:
        """Detect hanging pieces (undefended or underdefended)"""
        hanging_pieces = set()
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                # Count attackers and defenders
                attackers = len(board.attackers(not piece.color, square))
                defenders = len(board.attackers(piece.color, square))
                
                # Piece is hanging if more attackers than defenders
                if attackers > 0 and attackers > defenders:
                    hanging_pieces.add(square)
        
        return hanging_pieces
    
    def detect_tactical_opportunities(self, board: chess.Board, move: chess.Move) -> List[str]:
        """Enhanced tactical detection based on V12.6 weakness analysis"""
        tactics = []
        
        # Make the move to analyze resulting position
        try:
            board.push(move)
            
            # Fork detection (multiple pieces attacked)
            attacking_piece = board.piece_at(move.to_square)
            if attacking_piece:
                attacked_squares = []
                high_value_attacks = 0
                
                for square in chess.SQUARES:
                    if board.is_attacked_by(attacking_piece.color, square):
                        target = board.piece_at(square)
                        if target and target.color != attacking_piece.color:
                            attacked_squares.append(square)
                            if target.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                                high_value_attacks += 1
                
                if len(attacked_squares) >= 2:
                    if high_value_attacks >= 1:
                        tactics.append("Tactical Fork (High Value)")
                    else:
                        tactics.append("Tactical Fork")
            
            # Pin detection (piece cannot move without exposing king/valuable piece)
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color != board.turn:  # Enemy pieces
                    # Check if this piece is pinned
                    if self._is_piece_pinned(board, square):
                        tactics.append("Tactical Pin")
                        break
            
            # Discovery attack detection
            if self._creates_discovery_attack(board, move):
                tactics.append("Discovery Attack")
            
            # Back-rank mate threats
            if self._creates_back_rank_threat(board):
                tactics.append("Back Rank Threat")
                
        except Exception as e:
            # If there's any issue with tactical detection, continue without it
            pass
        finally:
            # Always pop the move if it was pushed
            if len(board.move_stack) > 0 and board.move_stack[-1] == move:
                board.pop()
        
        return tactics
    
    def _is_piece_pinned(self, board: chess.Board, square: chess.Square) -> bool:
        """Check if a piece is pinned"""
        piece = board.piece_at(square)
        if not piece:
            return False
        
        # Temporarily remove the piece and see if king is in check
        board.remove_piece_at(square)
        king_square = board.king(piece.color)
        is_pinned = bool(king_square and board.is_attacked_by(not piece.color, king_square))
        board.set_piece_at(square, piece)
        
        return is_pinned
    
    def _creates_discovery_attack(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move creates a discovery attack"""
        # This is a simplified version - would need more sophisticated analysis
        return False  # Placeholder
    
    def _creates_back_rank_threat(self, board: chess.Board) -> bool:
        """Check for back-rank mate threats"""
        # Simplified check for back-rank patterns
        return False  # Placeholder
    
    def enhanced_categorize_move(self, board: chess.Board, move: chess.Move) -> Tuple[List[str], int]:
        """Enhanced move categorization with V13.x improvements"""
        categories = []
        tactical_value = 0
        
        # Detect hanging pieces before move
        hanging_before = self.detect_hanging_pieces(board)
        
        # Basic move properties with enhanced analysis
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            
            if victim and attacker:
                victim_value = self.piece_values.get(victim.piece_type, 0)
                attacker_value = self.piece_values.get(attacker.piece_type, 0)
                
                # Enhanced capture analysis
                if victim_value > attacker_value:
                    categories.append("Good Capture")
                    tactical_value += 300
                elif victim_value == attacker_value:
                    categories.append("Equal Capture") 
                    tactical_value += 100
                else:
                    categories.append("Bad Capture")
                    tactical_value -= 50
                
                # MVV-LVA priority
                mvv_lva_score = victim_value - (attacker_value // 10)
                tactical_value += mvv_lva_score // 10
        
        # Check for piece safety improvements (Critical from weakness analysis!)
        if move.from_square in hanging_before:
            categories.append("Piece Safety (Save Hanging)")
            tactical_value += 500  # High priority!
        
        # Enhanced tactical detection
        tactical_opportunities = self.detect_tactical_opportunities(board, move)
        categories.extend(tactical_opportunities)
        tactical_value += len(tactical_opportunities) * 200
        
        # Check analysis
        board.push(move)
        if board.is_check():
            categories.append("Check")
            tactical_value += 150
            
            if board.is_checkmate():
                categories.append("Checkmate")
                tactical_value += 10000
                
        board.pop()
        
        # Development and positional factors
        piece = board.piece_at(move.from_square)
        if piece:
            # Piece development (important in opening)
            if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                start_rank = chess.square_rank(move.from_square)
                if ((piece.color == chess.WHITE and start_rank == 0) or 
                    (piece.color == chess.BLACK and start_rank == 7)):
                    categories.append("Piece Development")
                    tactical_value += 50
            
            # King safety (castling)
            if piece.piece_type == chess.KING:
                if board.is_castling(move):
                    categories.append("Castle")
                    tactical_value += 100
                else:
                    # Check if king move exposes it to danger
                    board.push(move)
                    if board.is_attacked_by(not piece.color, move.to_square):
                        categories.append("King Exposure")
                        tactical_value -= 200
                    board.pop()
        
        # Promotion analysis
        if move.promotion:
            if move.promotion == chess.QUEEN:
                categories.append("Queen Promotion")
                tactical_value += 800
            else:
                categories.append(f"Promotion to {chess.piece_name(move.promotion)}")
                tactical_value += 300
        
        # Special moves
        if board.is_en_passant(move):
            categories.append("En Passant")
            tactical_value += 100
        
        # If no specific categories, it's a quiet move
        if not categories:
            categories.append("Quiet Move")
            
        return categories, tactical_value
    
    def analyze_move_ordering_v13(self, board: chess.Board, max_depth: int = 5) -> Dict:
        """V13.x enhanced move ordering analysis"""
        legal_moves = list(board.legal_moves)
        stats = MoveOrderingStats(total_legal_moves=len(legal_moves))
        
        # Categorize all moves with enhanced system
        move_data = []
        move_categories = defaultdict(lambda: EnhancedMoveCategory("", 5))
        
        print(f"🔍 V13.x Enhanced Analysis: {len(legal_moves)} legal moves")
        
        for move in legal_moves:
            categories, tactical_value = self.enhanced_categorize_move(board, move)
            
            # Determine priority level (1=highest, 5=lowest)
            priority = 5  # Default lowest
            for category in categories:
                cat_priority = self.move_priorities.get(category, 5)
                priority = min(priority, cat_priority)  # Take highest priority
            
            move_info = {
                'move': move,
                'categories': categories,
                'tactical_value': tactical_value,
                'priority': priority,
                'uci': move.uci(),
                'san': board.san(move)
            }
            move_data.append(move_info)
            
            # Update statistics
            if 'Check' in categories or any('Tactical' in cat for cat in categories):
                stats.tactical_moves += 1
            if 'Piece Safety' in str(categories):
                stats.hanging_piece_saves += 1
            if any('Capture' in cat for cat in categories):
                stats.captures += 1
            if 'Check' in categories:
                stats.checks += 1
            if priority == 1:
                stats.high_priority_moves += 1
            elif priority == 2:
                stats.medium_priority_moves += 1
            elif priority >= 4:
                stats.low_priority_moves += 1
            
            # Add to categories
            for category in categories:
                if not move_categories[category].name:
                    move_categories[category].name = category
                    move_categories[category].priority_level = self.move_priorities.get(category, 5)
                
                move_desc = f"{move.uci()} ({board.san(move)})"
                move_categories[category].add_move(move, move_desc, 0, tactical_value)
        
        # Calculate search optimization potential
        high_med_count = stats.high_priority_moves + stats.medium_priority_moves
        if len(legal_moves) > 0:
            stats.search_reduction_potential = (1 - (high_med_count / len(legal_moves))) * 100
        
        # Sort moves by V13.x enhanced ordering
        optimal_ordering = sorted(move_data, key=lambda x: (x['priority'], -x['tactical_value']))
        
        # Get engine ordering for comparison
        engine_ordering = self._get_engine_ordering_v13(board)
        
        return {
            'stats': stats,
            'move_categories': dict(move_categories),
            'move_data': move_data,
            'optimal_ordering': optimal_ordering,
            'engine_ordering': engine_ordering,
            'legal_moves': legal_moves
        }
    
    def _get_engine_ordering_v13(self, board: chess.Board) -> List[Dict]:
        """Get engine move ordering with V13.x analysis"""
        if not self.engine:
            return self._fallback_ordering_v13(board)
        
        try:
            # Try to get engine's move ordering
            if hasattr(self.engine, '_order_moves'):
                engine_moves = self.engine._order_moves(board, list(board.legal_moves))
            else:
                engine_moves = self._fallback_ordering_v13(board)
            
            # Convert to our enhanced format
            engine_data = []
            for i, move in enumerate(engine_moves):
                if isinstance(move, dict):
                    # If already in dict format
                    move_obj = move.get('move', move)
                else:
                    # If it's a chess.Move object
                    move_obj = move
                
                engine_data.append({
                    'move': move_obj,
                    'engine_rank': i + 1,
                    'uci': move_obj.uci() if hasattr(move_obj, 'uci') else str(move_obj),
                    'san': board.san(move_obj) if hasattr(move_obj, 'uci') else str(move_obj)
                })
            
            return engine_data
            
        except Exception as e:
            print(f"Warning: Engine ordering failed: {e}")
            return self._fallback_ordering_v13(board)
    
    def _fallback_ordering_v13(self, board: chess.Board) -> List[Dict]:
        """Enhanced fallback ordering for V13.x"""
        moves = list(board.legal_moves)
        
        def enhanced_move_score(move):
            categories, tactical_value = self.enhanced_categorize_move(board, move)
            
            # Base score from tactical value
            score = tactical_value
            
            # Additional scoring based on V12.6 weakness analysis
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    victim_value = self.piece_values.get(victim.piece_type, 0)
                    attacker_value = self.piece_values.get(attacker.piece_type, 0)
                    # MVV-LVA with V13.x enhancement
                    score += (victim_value * 10 - attacker_value)
            
            # Prioritize moves that save hanging pieces (major weakness in V12.6)
            if 'Piece Safety' in str(categories):
                score += 1000
            
            # Prioritize tactical moves (70% missed in V12.6)
            if any('Tactical' in cat for cat in categories):
                score += 500
            
            return score
        
        moves.sort(key=enhanced_move_score, reverse=True)
        
        return [{
            'move': move,
            'engine_rank': i + 1,
            'uci': move.uci(),
            'san': board.san(move)
        } for i, move in enumerate(moves)]
    
    def print_v13_analysis(self, analysis: Dict, position_name: str = "Position"):
        """Print V13.x enhanced move ordering analysis"""
        print(f"\n{'='*80}")
        print(f"🚀 V13.x ENHANCED MOVE ORDERING ANALYSIS: {position_name}")
        print(f"{'='*80}")
        
        stats = analysis['stats']
        move_categories = analysis['move_categories']
        optimal_ordering = analysis['optimal_ordering']
        engine_ordering = analysis['engine_ordering']
        
        # V13.x Statistics Summary
        print(f"\n📊 V13.x PERFORMANCE METRICS:")
        print(f"  Total legal moves: {stats.total_legal_moves}")
        print(f"  High priority moves: {stats.high_priority_moves}")
        print(f"  Medium priority moves: {stats.medium_priority_moves}")
        print(f"  Low priority moves: {stats.low_priority_moves}")
        print(f"  Tactical moves detected: {stats.tactical_moves}")
        print(f"  Hanging piece saves: {stats.hanging_piece_saves}")
        print(f"  Search reduction potential: {stats.search_reduction_potential:.1f}%")
        
        # Priority-based categorization
        print(f"\n🎯 MOVE CATEGORIES BY PRIORITY:")
        print("-" * 60)
        
        priority_groups = defaultdict(list)
        for category_name, category in move_categories.items():
            priority_groups[category.priority_level].append((category_name, category))
        
        for priority in sorted(priority_groups.keys()):
            priority_name = {1: "CRITICAL", 2: "HIGH", 3: "MEDIUM", 4: "LOW", 5: "MINIMAL"}[priority]
            print(f"\n  🔥 PRIORITY {priority} ({priority_name}):")
            
            for category_name, category in sorted(priority_groups[priority], key=lambda x: len(x[1].moves), reverse=True):
                avg_tactical = sum(category.tactical_values) / len(category.tactical_values) if category.tactical_values else 0
                print(f"    {category_name}: {len(category)} moves (avg tactical value: {avg_tactical:.0f})")
                
                # Show top moves in each category
                for i, (move, desc) in enumerate(zip(category.moves[:2], category.descriptions[:2])):
                    tactical_val = category.tactical_values[i] if i < len(category.tactical_values) else 0
                    print(f"      • {desc} (tactical: {tactical_val})")
                
                if len(category) > 2:
                    print(f"      ... and {len(category) - 2} more")
        
        # V13.x Optimal vs Engine Ordering Comparison
        print(f"\n⚔️  V13.x OPTIMAL vs ENGINE ORDERING:")
        print("-" * 60)
        print("  Rank | V13.x Optimal        | Engine Choice        | Match?")
        print("-" * 60)
        
        for i in range(min(10, len(optimal_ordering), len(engine_ordering))):
            optimal_move = optimal_ordering[i]
            engine_move = engine_ordering[i] if i < len(engine_ordering) else None
            
            optimal_str = f"{optimal_move['san']} (P{optimal_move['priority']}, T{optimal_move['tactical_value']})"
            engine_str = f"{engine_move['san']}" if engine_move else "N/A"
            
            match = "✅" if engine_move and optimal_move['move'] == engine_move['move'] else "❌"
            
            print(f"  {i+1:2d}   | {optimal_str:20s} | {engine_str:20s} | {match}")
        
        # Critical Issues Detection
        print(f"\n🚨 V13.x CRITICAL ISSUES DETECTED:")
        print("-" * 60)
        
        issues = []
        
        # Check if engine is missing high-priority moves
        engine_top5 = [em['move'] for em in engine_ordering[:5]] if len(engine_ordering) >= 5 else []
        optimal_top5 = [om['move'] for om in optimal_ordering[:5]] if len(optimal_ordering) >= 5 else []
        
        missed_critical = [move for move in optimal_top5 if move not in engine_top5]
        if missed_critical:
            issues.append(f"❌ Engine missed {len(missed_critical)} critical moves in top 5")
        
        # Check for hanging piece detection
        if stats.hanging_piece_saves > 0:
            hanging_in_top5 = sum(1 for om in optimal_ordering[:5] if 'Piece Safety' in str(om['categories']))
            if hanging_in_top5 == 0:
                issues.append(f"❌ Engine didn't prioritize {stats.hanging_piece_saves} hanging piece saves")
        
        # Check tactical prioritization
        tactical_in_top5 = sum(1 for om in optimal_ordering[:5] if om['tactical_value'] > 200)
        if tactical_in_top5 > 0:
            engine_tactical_top5 = sum(1 for em in engine_ordering[:5] if em['move'] in [om['move'] for om in optimal_ordering if om['tactical_value'] > 200])
            if engine_tactical_top5 < tactical_in_top5:
                issues.append(f"❌ Engine missed {tactical_in_top5 - engine_tactical_top5} tactical moves in top 5")
        
        if issues:
            for issue in issues:
                print(f"  {issue}")
        else:
            print("  ✅ No critical V13.x issues detected!")
        
        # Performance recommendations
        print(f"\n💡 V13.x PERFORMANCE RECOMMENDATIONS:")
        print("-" * 60)
        
        recommendations = []
        
        if stats.search_reduction_potential > 50:
            recommendations.append(f"🎯 Focus search on top {stats.high_priority_moves + stats.medium_priority_moves} moves ({stats.search_reduction_potential:.0f}% reduction possible)")
        
        if stats.hanging_piece_saves > 0:
            recommendations.append(f"🔍 Prioritize {stats.hanging_piece_saves} piece safety moves earlier")
        
        if stats.tactical_moves > stats.high_priority_moves:
            recommendations.append(f"⚡ Enhance tactical detection - {stats.tactical_moves} tactical opportunities found")
        
        if not recommendations:
            recommendations.append("✅ Move ordering appears well-optimized for V13.x")
        
        for rec in recommendations:
            print(f"  {rec}")
    
    def test_weakness_positions(self):
        """Test move ordering on positions similar to V12.6 weaknesses"""
        weakness_positions = {
            "Hanging Piece Miss": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
            "Tactical Miss": "r2qkb1r/pp2nppp/3p1n2/2pP4/2P1P3/2N2N2/PP3PPP/R1BQKB1R w KQq - 0 6",
            "Move Ordering Test": "rnbqkb1r/pp1ppppp/5n2/2p5/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 2 3",
            "Complex Middlegame": "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NP1/PPP1NPB1/R1BQ1RK1 b - - 0 9",
            "Endgame Technique": "6k1/8/6K1/8/8/8/r7/8 w - - 0 1"
        }
        
        print(f"\n🔬 TESTING V13.x ON WEAKNESS-SIMILAR POSITIONS")
        print("=" * 80)
        
        for position_name, fen in weakness_positions.items():
            board = chess.Board(fen)
            analysis = self.analyze_move_ordering_v13(board)
            self.print_v13_analysis(analysis, position_name)
            print("\n" + "="*40 + "\n")


def main():
    """Run V13.x enhanced move ordering analysis"""
    print("🚀 V7P3R V13.x Enhanced Move Ordering Analysis")
    print("Fixing V12.6 weaknesses: 75% bad move ordering, 70% tactical misses")
    print("=" * 80)
    
    analyzer = EnhancedV7P3RMoveAnalyzer()
    
    # Test on positions that caused problems in V12.6
    analyzer.test_weakness_positions()
    
    print(f"\n📈 V13.x ANALYSIS COMPLETE")
    print("Ready for move ordering implementation!")


if __name__ == "__main__":
    main()