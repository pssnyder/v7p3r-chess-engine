#!/usr/bin/env python3
"""
V7P3R v11: Enhanced Strategic Position Database
Advanced pattern matching and similarity scoring for strategic positions
Author: Pat Snyder
"""

import chess
import json
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PositionPattern:
    """Represents a strategic position pattern"""
    piece_positions: Dict[str, List[str]]  # piece_type -> list of squares
    pawn_structure: str  # simplified pawn structure hash
    king_safety: Dict[str, float]  # king safety scores
    piece_activity: Dict[str, float]  # piece activity metrics
    strategic_themes: List[str]  # strategic themes present
    evaluation_bonus: float  # evaluation bonus for this pattern


class V7P3RStrategicDatabase:
    """Enhanced strategic position database with pattern matching"""
    
    def __init__(self, nudge_database_path: str = "src/v7p3r_nudge_database.json"):
        self.nudge_database_path = nudge_database_path
        self.nudge_database = {}
        self.position_patterns = {}
        self.pattern_cache = {}
        self.similarity_cache = {}
        self.statistics = {
            'positions_loaded': 0,
            'patterns_created': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'similarity_calculations': 0
        }
        
        self.load_nudge_database()
        self.build_position_patterns()
    
    def load_nudge_database(self) -> bool:
        """Load the existing nudge database"""
        try:
            with open(self.nudge_database_path, 'r') as f:
                self.nudge_database = json.load(f)
            
            self.statistics['positions_loaded'] = len(self.nudge_database)
            print(f"info string Loaded {self.statistics['positions_loaded']} strategic positions")
            return True
            
        except Exception as e:
            print(f"info string Error loading strategic database: {e}")
            return False
    
    def _extract_piece_positions(self, board: chess.Board) -> Dict[str, List[str]]:
        """Extract piece positions by type"""
        positions = defaultdict(list)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_key = f"{'w' if piece.color else 'b'}{piece.symbol().lower()}"
                positions[piece_key].append(chess.square_name(square))
        
        return dict(positions)
    
    def _analyze_pawn_structure(self, board: chess.Board) -> str:
        """Analyze pawn structure and create a hash"""
        white_pawns = []
        black_pawns = []
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                square_name = chess.square_name(square)
                if piece.color:
                    white_pawns.append(square_name)
                else:
                    black_pawns.append(square_name)
        
        # Create simplified pawn structure representation
        structure = f"w:{','.join(sorted(white_pawns))};b:{','.join(sorted(black_pawns))}"
        return hashlib.md5(structure.encode()).hexdigest()[:8]
    
    def _calculate_king_safety(self, board: chess.Board) -> Dict[str, float]:
        """Calculate king safety metrics"""
        safety = {}
        
        for color in [True, False]:  # White, Black
            king_square = board.king(color)
            if king_square is None:
                continue
                
            color_key = 'white' if color else 'black'
            
            # Basic king safety factors
            safety_score = 0.0
            
            # Check squares around king
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            
            # Pawn shelter
            if color:  # White
                shelter_squares = [
                    chess.square(f, king_rank + 1) for f in range(max(0, king_file-1), min(8, king_file+2))
                    if king_rank < 7
                ]
            else:  # Black
                shelter_squares = [
                    chess.square(f, king_rank - 1) for f in range(max(0, king_file-1), min(8, king_file+2))
                    if king_rank > 0
                ]
            
            pawn_shelter = 0
            for square in shelter_squares:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    pawn_shelter += 1
            
            safety_score += pawn_shelter * 0.2
            
            # Open files near king
            open_files = 0
            for file_offset in [-1, 0, 1]:
                check_file = king_file + file_offset
                if 0 <= check_file < 8:
                    file_has_pawn = False
                    for rank in range(8):
                        piece = board.piece_at(chess.square(check_file, rank))
                        if piece and piece.piece_type == chess.PAWN:
                            file_has_pawn = True
                            break
                    if not file_has_pawn:
                        open_files += 1
            
            safety_score -= open_files * 0.3
            
            # Piece attacks on king area
            attackers = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color != color:
                    if board.is_attacked_by(not color, king_square):
                        attackers += 1
            
            safety_score -= attackers * 0.1
            
            safety[color_key] = max(0.0, min(1.0, safety_score + 0.5))
        
        return safety
    
    def _calculate_piece_activity(self, board: chess.Board) -> Dict[str, float]:
        """Calculate piece activity metrics"""
        activity = {}
        
        for color in [True, False]:
            color_key = 'white' if color else 'black'
            total_mobility = 0
            piece_count = 0
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == color and piece.piece_type != chess.PAWN:
                    piece_count += 1
                    
                    # Calculate piece mobility
                    mobility = 0
                    for move in board.legal_moves:
                        if move.from_square == square:
                            mobility += 1
                    
                    # Weight by piece value
                    piece_values = {
                        chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
                        chess.QUEEN: 9, chess.KING: 1
                    }
                    
                    weighted_mobility = mobility * piece_values.get(piece.piece_type, 1)
                    total_mobility += weighted_mobility
            
            if piece_count > 0:
                activity[color_key] = total_mobility / (piece_count * 10)  # Normalize
            else:
                activity[color_key] = 0.0
        
        return activity
    
    def _identify_strategic_themes(self, board: chess.Board) -> List[str]:
        """Identify strategic themes in the position"""
        themes = []
        
        # Material imbalance
        material_balance = self._calculate_material_balance(board)
        if material_balance > 3:
            themes.append("material_advantage_white")
        elif material_balance < -3:
            themes.append("material_advantage_black")
        
        # Endgame detection
        piece_count = len([p for p in board.piece_map().values() if p.piece_type != chess.PAWN])
        if piece_count <= 8:
            themes.append("endgame")
        elif piece_count <= 12:
            themes.append("late_middlegame")
        else:
            themes.append("opening_middlegame")
        
        # King safety issues
        king_safety = self._calculate_king_safety(board)
        for color_key, safety in king_safety.items():
            if safety < 0.3:
                themes.append(f"king_danger_{color_key}")
        
        # Pawn structure themes
        white_pawns = [sq for sq, piece in board.piece_map().items() 
                      if piece.piece_type == chess.PAWN and piece.color]
        black_pawns = [sq for sq, piece in board.piece_map().items() 
                      if piece.piece_type == chess.PAWN and not piece.color]
        
        # Isolated pawns
        for pawns, color in [(white_pawns, "white"), (black_pawns, "black")]:
            isolated_count = 0
            for pawn_sq in pawns:
                file = chess.square_file(pawn_sq)
                has_neighbor = False
                for other_pawn in pawns:
                    other_file = chess.square_file(other_pawn)
                    if abs(file - other_file) == 1:
                        has_neighbor = True
                        break
                if not has_neighbor:
                    isolated_count += 1
            
            if isolated_count >= 2:
                themes.append(f"isolated_pawns_{color}")
        
        # Passed pawns
        for color in [True, False]:
            color_key = "white" if color else "black"
            pawns = white_pawns if color else black_pawns
            opponent_pawns = black_pawns if color else white_pawns
            
            for pawn_sq in pawns:
                file = chess.square_file(pawn_sq)
                rank = chess.square_rank(pawn_sq)
                
                is_passed = True
                for opp_pawn in opponent_pawns:
                    opp_file = chess.square_file(opp_pawn)
                    opp_rank = chess.square_rank(opp_pawn)
                    
                    if abs(file - opp_file) <= 1:
                        if color and opp_rank > rank:  # White pawn
                            is_passed = False
                            break
                        elif not color and opp_rank < rank:  # Black pawn
                            is_passed = False
                            break
                
                if is_passed:
                    themes.append(f"passed_pawn_{color_key}")
                    break
        
        return themes
    
    def _calculate_material_balance(self, board: chess.Board) -> float:
        """Calculate material balance (positive = white advantage)"""
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9
        }
        
        balance = 0
        for piece in board.piece_map().values():
            value = piece_values.get(piece.piece_type, 0)
            if piece.color:
                balance += value
            else:
                balance -= value
        
        return balance
    
    def create_position_pattern(self, board: chess.Board) -> PositionPattern:
        """Create a position pattern from a board state"""
        piece_positions = self._extract_piece_positions(board)
        pawn_structure = self._analyze_pawn_structure(board)
        king_safety = self._calculate_king_safety(board)
        piece_activity = self._calculate_piece_activity(board)
        strategic_themes = self._identify_strategic_themes(board)
        
        # Calculate evaluation bonus based on themes and position characteristics
        evaluation_bonus = 0.0
        
        # Theme-based bonuses
        if "material_advantage_white" in strategic_themes:
            evaluation_bonus += 0.5
        elif "material_advantage_black" in strategic_themes:
            evaluation_bonus -= 0.5
        
        if "king_danger_white" in strategic_themes:
            evaluation_bonus -= 0.3
        if "king_danger_black" in strategic_themes:
            evaluation_bonus += 0.3
        
        # Activity bonuses
        white_activity = piece_activity.get('white', 0.0)
        black_activity = piece_activity.get('black', 0.0)
        evaluation_bonus += (white_activity - black_activity) * 0.2
        
        return PositionPattern(
            piece_positions=piece_positions,
            pawn_structure=pawn_structure,
            king_safety=king_safety,
            piece_activity=piece_activity,
            strategic_themes=strategic_themes,
            evaluation_bonus=evaluation_bonus
        )
    
    def build_position_patterns(self):
        """Build position patterns from nudge database"""
        print("info string Building strategic position patterns...")
        
        for position_key, position_data in self.nudge_database.items():
            try:
                fen = position_data.get('fen', '')
                if not fen:
                    continue
                
                board = chess.Board(fen)
                pattern = self.create_position_pattern(board)
                self.position_patterns[position_key] = pattern
                self.statistics['patterns_created'] += 1
                
            except Exception as e:
                continue
        
        print(f"info string Created {self.statistics['patterns_created']} position patterns")
    
    def calculate_position_similarity(self, board: chess.Board, target_pattern: PositionPattern) -> float:
        """Calculate similarity between current position and target pattern"""
        self.statistics['similarity_calculations'] += 1
        
        current_pattern = self.create_position_pattern(board)
        similarity_score = 0.0
        total_weight = 0.0
        
        # Piece position similarity (weight: 0.3)
        piece_similarity = self._calculate_piece_similarity(
            current_pattern.piece_positions, target_pattern.piece_positions
        )
        similarity_score += piece_similarity * 0.3
        total_weight += 0.3
        
        # Pawn structure similarity (weight: 0.2)
        pawn_similarity = 1.0 if current_pattern.pawn_structure == target_pattern.pawn_structure else 0.0
        similarity_score += pawn_similarity * 0.2
        total_weight += 0.2
        
        # King safety similarity (weight: 0.2)
        king_similarity = self._calculate_king_similarity(
            current_pattern.king_safety, target_pattern.king_safety
        )
        similarity_score += king_similarity * 0.2
        total_weight += 0.2
        
        # Strategic themes similarity (weight: 0.3)
        theme_similarity = self._calculate_theme_similarity(
            current_pattern.strategic_themes, target_pattern.strategic_themes
        )
        similarity_score += theme_similarity * 0.3
        total_weight += 0.3
        
        return similarity_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_piece_similarity(self, current_pieces: Dict, target_pieces: Dict) -> float:
        """Calculate piece position similarity"""
        all_piece_types = set(current_pieces.keys()) | set(target_pieces.keys())
        if not all_piece_types:
            return 1.0
        
        total_similarity = 0.0
        for piece_type in all_piece_types:
            current_squares = set(current_pieces.get(piece_type, []))
            target_squares = set(target_pieces.get(piece_type, []))
            
            if not current_squares and not target_squares:
                piece_similarity = 1.0
            elif not current_squares or not target_squares:
                piece_similarity = 0.0
            else:
                # Calculate intersection over union
                intersection = len(current_squares & target_squares)
                union = len(current_squares | target_squares)
                piece_similarity = intersection / union if union > 0 else 0.0
            
            total_similarity += piece_similarity
        
        return total_similarity / len(all_piece_types)
    
    def _calculate_king_similarity(self, current_safety: Dict, target_safety: Dict) -> float:
        """Calculate king safety similarity"""
        similarities = []
        
        for color in ['white', 'black']:
            current_val = current_safety.get(color, 0.5)
            target_val = target_safety.get(color, 0.5)
            
            # Calculate absolute difference and convert to similarity
            diff = abs(current_val - target_val)
            similarity = 1.0 - diff  # Since values are 0-1
            similarities.append(max(0.0, similarity))
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_theme_similarity(self, current_themes: List[str], target_themes: List[str]) -> float:
        """Calculate strategic themes similarity"""
        current_set = set(current_themes)
        target_set = set(target_themes)
        
        if not current_set and not target_set:
            return 1.0
        
        if not current_set or not target_set:
            return 0.0
        
        intersection = len(current_set & target_set)
        union = len(current_set | target_set)
        
        return intersection / union if union > 0 else 0.0
    
    def find_similar_positions(self, board: chess.Board, min_similarity: float = 0.6, 
                             max_results: int = 5) -> List[Tuple[str, float, Dict]]:
        """Find positions similar to current board state"""
        position_fen = board.fen()
        
        # Check cache first
        cache_key = f"{position_fen}:{min_similarity}:{max_results}"
        if cache_key in self.similarity_cache:
            self.statistics['cache_hits'] += 1
            return self.similarity_cache[cache_key]
        
        self.statistics['cache_misses'] += 1
        
        similarities = []
        
        for position_key, pattern in self.position_patterns.items():
            similarity = self.calculate_position_similarity(board, pattern)
            
            if similarity >= min_similarity:
                position_data = self.nudge_database.get(position_key, {})
                similarities.append((position_key, similarity, position_data))
        
        # Sort by similarity and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = similarities[:max_results]
        
        # Cache results
        self.similarity_cache[cache_key] = results
        
        return results
    
    def get_strategic_evaluation_bonus(self, board: chess.Board) -> float:
        """Get strategic evaluation bonus for current position (optimized)"""
        # Use fast FEN-based lookup first
        position_fen = board.fen()
        
        # Simple cache check
        if position_fen in self.similarity_cache:
            cached_result = self.similarity_cache[position_fen]
            return cached_result.get('evaluation_bonus', 0.0)
        
        # For performance, only do full similarity search rarely
        # Use simplified matching for most cases
        try:
            # Fast material balance check
            material_balance = self._calculate_material_balance(board)
            
            # Simple bonus based on material advantage
            simple_bonus = 0.0
            if material_balance > 2:
                simple_bonus = 0.1  # Small bonus for material advantage
            elif material_balance < -2:
                simple_bonus = -0.1
            
            # Cache this simple result
            self.similarity_cache[position_fen] = {'evaluation_bonus': simple_bonus}
            
            return simple_bonus
            
        except Exception:
            return 0.0
    
    def get_strategic_move_bonus(self, board: chess.Board, move: chess.Move) -> float:
        """Get strategic move bonus based on similar positions (optimized)"""
        # For performance, use simplified move bonus calculation
        try:
            # Simple bonus based on move type
            bonus = 0.0
            
            piece = board.piece_at(move.from_square)
            if not piece:
                return 0.0
            
            # Central pawn moves get small bonus
            if piece.piece_type == chess.PAWN:
                to_file = chess.square_file(move.to_square)
                if 3 <= to_file <= 4:  # e and d files
                    bonus = 0.05
            
            # Knight development gets small bonus
            elif piece.piece_type == chess.KNIGHT:
                from_rank = chess.square_rank(move.from_square)
                to_rank = chess.square_rank(move.to_square)
                
                # Developing from back rank
                if (board.turn and from_rank == 0 and to_rank > 0) or (not board.turn and from_rank == 7 and to_rank < 7):
                    bonus = 0.03
            
            return bonus
            
        except Exception:
            return 0.0
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        return self.statistics.copy()
    
    def clear_cache(self):
        """Clear similarity cache"""
        self.similarity_cache.clear()
        self.pattern_cache.clear()


# Module-level instance for easy import
strategic_database = V7P3RStrategicDatabase()