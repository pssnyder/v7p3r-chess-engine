#!/usr/bin/env python3
"""
V7P3R Intelligent Nudge System v2.0
=====================================

A performance-optimized nudge system that subtly influences evaluation through:
1. Enhanced piece-square tables based on move frequency data
2. Opening book integration for better center control
3. Light move ordering adjustments
4. Gradual learning from historical game data

This replaces the heavy runtime nudge system with pre-computed adjustments.
"""

import json
import chess
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hashlib

@dataclass
class NudgeStats:
    """Statistics for a nudge move"""
    frequency: int
    confidence: float
    eval_score: float
    classification: str
    themes: List[str]

class V7P3RIntelligentNudges:
    """Intelligent nudge system with pre-computed optimizations"""
    
    def __init__(self, nudge_db_path: str = "src/v7p3r_enhanced_nudges.json"):
        self.nudge_db_path = nudge_db_path
        self.nudge_data = {}
        self.piece_square_adjustments = {}
        self.opening_preferences = {}
        self.move_ordering_bonuses = {}
        
        # Performance settings
        self.max_positions_analyzed = 5000  # Limit for performance
        self.min_confidence = 0.3  # Only trust moves with decent confidence
        self.min_frequency = 2  # Moves must appear at least twice
        
        # Load and process nudge data
        self._load_nudge_database()
        self._analyze_opening_preferences()
        self._generate_piece_square_adjustments()
        self._compute_move_ordering_bonuses()
        
        print(f"🧠 V7P3R Intelligent Nudges initialized:")
        print(f"   📊 Analyzed {len(self.nudge_data)} positions")
        print(f"   🏰 Opening preferences: {len(self.opening_preferences)} moves")
        print(f"   📋 Piece square adjustments: {len(self.piece_square_adjustments)} squares")
        print(f"   🎯 Move ordering bonuses: {len(self.move_ordering_bonuses)} moves")
    
    def _load_nudge_database(self):
        """Load nudge database with performance limits"""
        try:
            with open(self.nudge_db_path, 'r') as f:
                raw_data = json.load(f)
            
            # Process limited number of positions for performance
            positions_processed = 0
            for pos_hash, pos_data in raw_data.items():
                if positions_processed >= self.max_positions_analyzed:
                    break
                
                fen = pos_data.get('fen', '')
                moves = pos_data.get('moves', {})
                
                # Filter high-quality moves
                quality_moves = {}
                for move, move_data in moves.items():
                    frequency = move_data.get('frequency', 0)
                    confidence = move_data.get('confidence', 0.0)
                    
                    if frequency >= self.min_frequency and confidence >= self.min_confidence:
                        quality_moves[move] = NudgeStats(
                            frequency=frequency,
                            confidence=confidence,
                            eval_score=move_data.get('eval', 0.0),
                            classification=move_data.get('tactical_info', {}).get('classification', 'unknown'),
                            themes=move_data.get('tactical_info', {}).get('themes', [])
                        )
                
                if quality_moves:
                    self.nudge_data[fen] = quality_moves
                    positions_processed += 1
                    
        except FileNotFoundError:
            print(f"⚠️  Nudge database not found: {self.nudge_db_path}")
        except Exception as e:
            print(f"⚠️  Error loading nudge database: {e}")
    
    def _analyze_opening_preferences(self):
        """Extract opening move preferences for better center control"""
        opening_moves = {}
        
        for fen, moves in self.nudge_data.items():
            # Parse FEN to get move number
            try:
                board = chess.Board(fen)
                move_number = board.fullmove_number
                
                # Focus on opening moves (first 8 moves)
                if move_number <= 8:
                    for move_uci, stats in moves.items():
                        if move_uci not in opening_moves:
                            opening_moves[move_uci] = {
                                'total_frequency': 0,
                                'total_confidence': 0.0,
                                'count': 0,
                                'avg_eval': 0.0,
                                'early_game_bonus': 0.0
                            }
                        
                        opening_moves[move_uci]['total_frequency'] += stats.frequency
                        opening_moves[move_uci]['total_confidence'] += stats.confidence
                        opening_moves[move_uci]['avg_eval'] += stats.eval_score
                        opening_moves[move_uci]['count'] += 1
                        
                        # Give bonus for very early moves
                        if move_number <= 3:
                            opening_moves[move_uci]['early_game_bonus'] += 10.0
                        elif move_number <= 6:
                            opening_moves[move_uci]['early_game_bonus'] += 5.0
            except:
                continue
        
        # Compute averages and filter good opening moves
        for move, data in opening_moves.items():
            if data['count'] >= 2:  # Appears in multiple games
                avg_confidence = data['total_confidence'] / data['count']
                avg_eval = data['avg_eval'] / data['count']
                
                if avg_confidence >= 0.4:  # Good confidence
                    bonus = min(20.0, data['early_game_bonus'] + avg_eval * 2)
                    self.opening_preferences[move] = bonus
        
        # Add specific Caro-Kann and center control moves
        center_control_moves = {
            'e2e4': 15.0,  # King's pawn
            'd2d4': 15.0,  # Queen's pawn
            'c7c6': 12.0,  # Caro-Kann preparation
            'e7e6': 10.0,  # French Defense
            'g1f3': 8.0,   # Knight development
            'b1c3': 8.0,   # Knight development
            'd7d5': 12.0,  # Central pawn advance
            'c2c4': 10.0,  # English Opening
        }
        
        for move, bonus in center_control_moves.items():
            if move not in self.opening_preferences:
                self.opening_preferences[move] = bonus
            else:
                self.opening_preferences[move] = max(self.opening_preferences[move], bonus)
    
    def _generate_piece_square_adjustments(self):
        """Generate piece-square table adjustments based on move frequency"""
        square_usage = {}
        
        for fen, moves in self.nudge_data.items():
            for move_uci, stats in moves.items():
                try:
                    move = chess.Move.from_uci(move_uci)
                    to_square = move.to_square
                    
                    # Weight by frequency and confidence
                    weight = stats.frequency * stats.confidence
                    
                    if to_square not in square_usage:
                        square_usage[to_square] = {
                            'total_weight': 0.0,
                            'pawn_weight': 0.0,
                            'knight_weight': 0.0,
                            'bishop_weight': 0.0,
                            'rook_weight': 0.0,
                            'queen_weight': 0.0,
                            'king_weight': 0.0
                        }
                    
                    square_usage[to_square]['total_weight'] += weight
                    
                    # Try to determine piece type from FEN context
                    try:
                        board = chess.Board(fen)
                        piece = board.piece_at(move.from_square)
                        if piece:
                            piece_name = chess.piece_name(piece.piece_type).lower()
                            square_usage[to_square][f'{piece_name}_weight'] += weight
                    except:
                        pass
                        
                except ValueError:
                    continue
        
        # Convert to piece-square adjustments
        for square, usage in square_usage.items():
            if usage['total_weight'] >= 3.0:  # Significant usage
                # Scale adjustment based on usage frequency
                base_adjustment = min(10.0, usage['total_weight'] * 1.5)
                
                self.piece_square_adjustments[square] = {
                    'pawn': base_adjustment * 0.5,
                    'knight': base_adjustment * 0.8,
                    'bishop': base_adjustment * 0.7,
                    'rook': base_adjustment * 0.6,
                    'queen': base_adjustment * 0.4,
                    'king': base_adjustment * 0.3
                }
        
        # Add center control bonuses
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        extended_center = [chess.C3, chess.C4, chess.C5, chess.C6,
                          chess.D3, chess.D6, chess.E3, chess.E6,
                          chess.F3, chess.F4, chess.F5, chess.F6]
        
        for square in center_squares:
            if square not in self.piece_square_adjustments:
                self.piece_square_adjustments[square] = {}
            adj = self.piece_square_adjustments[square]
            adj['pawn'] = adj.get('pawn', 0) + 15.0
            adj['knight'] = adj.get('knight', 0) + 12.0
            adj['bishop'] = adj.get('bishop', 0) + 8.0
        
        for square in extended_center:
            if square not in self.piece_square_adjustments:
                self.piece_square_adjustments[square] = {}
            adj = self.piece_square_adjustments[square]
            adj['pawn'] = adj.get('pawn', 0) + 8.0
            adj['knight'] = adj.get('knight', 0) + 6.0
            adj['bishop'] = adj.get('bishop', 0) + 4.0
    
    def _compute_move_ordering_bonuses(self):
        """Compute move ordering bonuses for high-frequency moves"""
        for fen, moves in self.nudge_data.items():
            for move_uci, stats in moves.items():
                if stats.frequency >= 3 and stats.confidence >= 0.5:
                    # Calculate bonus based on frequency, confidence, and eval
                    bonus = min(15.0, stats.frequency * 2 + stats.confidence * 10 + max(0, stats.eval_score))
                    
                    # Add theme-based bonuses
                    if 'mate' in stats.themes:
                        bonus += 20.0
                    elif 'attack' in ' '.join(stats.themes):
                        bonus += 10.0
                    elif 'development' in stats.classification:
                        bonus += 5.0
                    
                    self.move_ordering_bonuses[move_uci] = bonus
    
    def get_opening_bonus(self, move_uci: str, move_number: int) -> float:
        """Get opening move bonus for early game center control"""
        if move_number > 8:
            return 0.0
        
        bonus = self.opening_preferences.get(move_uci, 0.0)
        
        # Scale bonus by move number (earlier moves get higher bonus)
        if move_number <= 3:
            return bonus
        elif move_number <= 6:
            return bonus * 0.7
        else:
            return bonus * 0.4
    
    def get_piece_square_adjustment(self, piece_type: chess.PieceType, square: int) -> float:
        """Get piece-square table adjustment for a specific piece and square"""
        if square not in self.piece_square_adjustments:
            return 0.0
        
        piece_name = chess.piece_name(piece_type).lower()
        return self.piece_square_adjustments[square].get(piece_name, 0.0)
    
    def get_move_ordering_bonus(self, move_uci: str) -> float:
        """Get move ordering bonus for known good moves"""
        return self.move_ordering_bonuses.get(move_uci, 0.0)
    
    def is_preferred_move(self, move_uci: str, fen: Optional[str] = None) -> Tuple[bool, float]:
        """Check if a move is preferred with confidence score"""
        # Check opening preferences
        try:
            board = chess.Board(fen) if fen else None
            if board and board.fullmove_number <= 8:
                bonus = self.get_opening_bonus(move_uci, board.fullmove_number)
                if bonus > 5.0:
                    return True, bonus
        except:
            pass
        
        # Check specific position database
        if fen and fen in self.nudge_data:
            if move_uci in self.nudge_data[fen]:
                stats = self.nudge_data[fen][move_uci]
                return True, stats.confidence * 20.0
        
        # Check move ordering bonuses
        bonus = self.get_move_ordering_bonus(move_uci)
        if bonus > 8.0:
            return True, bonus
        
        return False, 0.0
    
    def get_enhanced_piece_square_tables(self, base_tables: Dict) -> Dict:
        """Enhance base piece-square tables with nudge adjustments"""
        enhanced_tables = {}
        
        for piece_name, base_table in base_tables.items():
            enhanced_table = base_table.copy()
            
            # Apply nudge adjustments
            for square, adjustments in self.piece_square_adjustments.items():
                if piece_name in adjustments:
                    adjustment = adjustments[piece_name]
                    
                    # Convert square to table index (flip for white/black perspective)
                    white_index = square
                    black_index = 63 - square
                    
                    # Apply adjustment (scale down to avoid overwhelming base values)
                    scaled_adjustment = adjustment * 0.3  # 30% influence
                    
                    if len(enhanced_table) == 64:  # Single table
                        enhanced_table[white_index] += scaled_adjustment
                    elif len(enhanced_table) == 2:  # White/Black tables
                        if isinstance(enhanced_table, dict) and 'white' in enhanced_table:
                            enhanced_table['white'][white_index] += scaled_adjustment
                            enhanced_table['black'][black_index] += scaled_adjustment
            
            enhanced_tables[piece_name] = enhanced_table
        
        return enhanced_tables

def analyze_nudge_system_impact():
    """Test function to analyze the impact of the nudge system"""
    nudges = V7P3RIntelligentNudges()
    
    print("\n🔍 NUDGE SYSTEM ANALYSIS")
    print("=" * 50)
    
    # Test opening preferences
    print("\n📋 Opening Preferences (Top 10):")
    sorted_openings = sorted(nudges.opening_preferences.items(), key=lambda x: x[1], reverse=True)[:10]
    for move, bonus in sorted_openings:
        print(f"  {move}: +{bonus:.1f}")
    
    # Test piece square adjustments
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    print(f"\n🎯 Center Square Adjustments:")
    for square in center_squares:
        square_name = chess.square_name(square)
        if square in nudges.piece_square_adjustments:
            adj = nudges.piece_square_adjustments[square]
            print(f"  {square_name}: Pawn +{adj.get('pawn', 0):.1f}, Knight +{adj.get('knight', 0):.1f}")
    
    # Test move ordering bonuses
    print(f"\n⚡ Move Ordering Bonuses (Top 10):")
    sorted_bonuses = sorted(nudges.move_ordering_bonuses.items(), key=lambda x: x[1], reverse=True)[:10]
    for move, bonus in sorted_bonuses:
        print(f"  {move}: +{bonus:.1f}")
    
    print(f"\n✅ Nudge system ready for integration!")

if __name__ == "__main__":
    analyze_nudge_system_impact()