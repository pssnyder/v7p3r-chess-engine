#!/usr/bin/env python3
"""
V7P3R Enhanced Move Orderer with Dynamic Positional Intelligence
Implements sophisticated move classification and adaptive search guidance
Author: Pat Snyder
"""

import chess
from typing import List, Tuple, Optional, Dict, Any, NamedTuple
from enum import Enum
from dataclasses import dataclass
import time


class MoveClass(Enum):
    """Move classification for tactical and positional understanding"""
    OFFENSIVE = "offensive"           # Captures, attacks, checks
    DEFENSIVE = "defensive"           # Piece defense, blocking, retreat
    DEVELOPMENTAL = "developmental"   # Piece development, castling, central control
    TACTICAL = "tactical"            # Complex tactical motifs (forks, pins, etc.)
    ENDGAME = "endgame"              # King activity, pawn promotion, centralization


@dataclass
class ClassifiedMove:
    """Move with classification and scoring data"""
    move: chess.Move
    primary_class: MoveClass
    secondary_class: Optional[MoveClass]
    score: int
    tactical_features: List[str]
    positional_features: List[str]


@dataclass 
class PositionTone:
    """Analysis of position's strategic character"""
    tactical_complexity: float      # 0.0 = quiet, 1.0 = highly tactical
    king_safety_urgency: float     # 0.0 = safe, 1.0 = critical danger
    development_priority: float    # 0.0 = developed, 1.0 = needs development  
    endgame_factor: float          # 0.0 = opening/middle, 1.0 = pure endgame
    aggression_viability: float    # 0.0 = defensive, 1.0 = attacking chances
    
    # Dynamic factors updated during search
    offensive_success_rate: float = 0.5
    defensive_success_rate: float = 0.5  
    developmental_success_rate: float = 0.5


class SearchFeedback:
    """Tracks move type performance during search to guide future decisions"""
    
    def __init__(self):
        self.move_type_attempts = {move_class: 0 for move_class in MoveClass}
        self.move_type_successes = {move_class: 0 for move_class in MoveClass}
        self.move_type_failures = {move_class: 0 for move_class in MoveClass}
        self.recent_evaluations = []
        self.position_understanding_confidence = 0.0
    
    def record_move_result(self, move_class: MoveClass, evaluation: float, improved: bool):
        """Record the result of trying a move of this type"""
        self.move_type_attempts[move_class] += 1
        if improved:
            self.move_type_successes[move_class] += 1
        else:
            self.move_type_failures[move_class] += 1
        
        self.recent_evaluations.append((move_class, evaluation, improved))
        
        # Keep only recent data
        if len(self.recent_evaluations) > 20:
            self.recent_evaluations.pop(0)
    
    def get_move_type_confidence(self, move_class: MoveClass) -> float:
        """Get confidence that this move type is working in current position"""
        attempts = self.move_type_attempts[move_class]
        if attempts == 0:
            return 0.5  # Neutral confidence
        
        successes = self.move_type_successes[move_class]
        return successes / attempts
    
    def should_deprioritize_move_type(self, move_class: MoveClass, threshold: float = 0.3) -> bool:
        """Determine if we should deprioritize this move type based on poor performance"""
        attempts = self.move_type_attempts[move_class]
        if attempts < 3:  # Need some data before making decisions
            return False
        
        confidence = self.get_move_type_confidence(move_class)
        return confidence < threshold
    
    def update_position_tone(self, position_tone: PositionTone):
        """Update position tone based on search feedback"""
        position_tone.offensive_success_rate = self.get_move_type_confidence(MoveClass.OFFENSIVE)
        position_tone.defensive_success_rate = self.get_move_type_confidence(MoveClass.DEFENSIVE)
        position_tone.developmental_success_rate = self.get_move_type_confidence(MoveClass.DEVELOPMENTAL)


class V7P3REnhancedMoveOrderer:
    """Enhanced move ordering with dynamic positional intelligence"""
    
    def __init__(self):
        # Basic piece values for MVV-LVA
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 10000
        }
        
        # Traditional move ordering components
        self.killer_moves = {}
        self.history_heuristic = {}
        
        # Enhanced components
        self.search_feedback = SearchFeedback()
        self.current_position_tone = None
        
        # Configuration
        self.max_move_classification_time = 0.01  # 10ms max for classification
        
    def analyze_position_tone(self, board: chess.Board) -> PositionTone:
        """Analyze the strategic character of the position"""
        start_time = time.time()
        
        # Material analysis
        white_material = self._calculate_material(board, chess.WHITE)
        black_material = self._calculate_material(board, chess.BLACK)
        total_material = white_material + black_material
        
        # Basic position metrics
        tactical_complexity = self._assess_tactical_complexity(board)
        king_safety_urgency = self._assess_king_safety(board)
        development_priority = self._assess_development_needs(board)
        endgame_factor = self._assess_endgame_factor(board, total_material)
        aggression_viability = self._assess_aggression_viability(board)
        
        tone = PositionTone(
            tactical_complexity=tactical_complexity,
            king_safety_urgency=king_safety_urgency, 
            development_priority=development_priority,
            endgame_factor=endgame_factor,
            aggression_viability=aggression_viability
        )
        
        # Apply search feedback if available
        if hasattr(self, 'search_feedback'):
            self.search_feedback.update_position_tone(tone)
        
        elapsed = time.time() - start_time
        if elapsed > self.max_move_classification_time:
            print(f"info string Position analysis took {elapsed:.3f}s")
        
        return tone
    
    def classify_and_order_moves(self, board: chess.Board, moves: List[chess.Move], 
                                depth: int = 0, tt_move: Optional[chess.Move] = None) -> List[ClassifiedMove]:
        """Main entry point: classify moves and order them with dynamic intelligence"""
        if not moves:
            return []
        
        # Analyze position tone if not already done
        if self.current_position_tone is None:
            self.current_position_tone = self.analyze_position_tone(board)
        
        # Classify all moves
        classified_moves = []
        for move in moves:
            classified_move = self._classify_move(board, move, depth, tt_move)
            classified_moves.append(classified_move)
        
        # Apply dynamic prioritization based on search feedback
        self._apply_dynamic_prioritization(classified_moves)
        
        # Sort by final score
        classified_moves.sort(key=lambda x: x.score, reverse=True)
        
        return classified_moves
    
    def order_moves(self, board: chess.Board, moves: List[chess.Move], 
                   depth: int = 0, tt_move: Optional[chess.Move] = None) -> List[chess.Move]:
        """Traditional interface - returns ordered moves without classification"""
        classified_moves = self.classify_and_order_moves(board, moves, depth, tt_move)
        return [cm.move for cm in classified_moves]
    
    def _classify_move(self, board: chess.Board, move: chess.Move, depth: int, 
                      tt_move: Optional[chess.Move]) -> ClassifiedMove:
        """Classify a single move and determine its characteristics"""
        
        # Start with base score from traditional ordering
        base_score = self._get_traditional_score(board, move, depth, tt_move)
        
        # Analyze move characteristics
        tactical_features = self._identify_tactical_features(board, move)
        positional_features = self._identify_positional_features(board, move)
        
        # Determine primary classification
        primary_class, secondary_class = self._determine_move_classes(
            board, move, tactical_features, positional_features
        )
        
        return ClassifiedMove(
            move=move,
            primary_class=primary_class,
            secondary_class=secondary_class,
            score=base_score,
            tactical_features=tactical_features,
            positional_features=positional_features
        )
    
    def _get_traditional_score(self, board: chess.Board, move: chess.Move, 
                             depth: int, tt_move: Optional[chess.Move]) -> int:
        """Get traditional move ordering score as baseline"""
        score = 0
        
        # 1. Transposition table move
        if tt_move and move == tt_move:
            return 10000
        
        # 2. Captures (MVV-LVA)
        if board.is_capture(move):
            victim_value = 0
            attacker_value = 0
            
            captured_piece = board.piece_at(move.to_square)
            moving_piece = board.piece_at(move.from_square)
            
            if captured_piece:
                victim_value = self.piece_values[captured_piece.piece_type]
            if moving_piece:
                attacker_value = self.piece_values[moving_piece.piece_type]
            
            mvv_lva_score = victim_value - (attacker_value // 10)
            
            if mvv_lva_score > 0:
                score = 8000 + mvv_lva_score
            else:
                score = 6000 + mvv_lva_score
        
        # 3. Checks
        elif board.gives_check(move):
            score = 7000
        
        # 4. Killer moves
        elif self._is_killer_move(move, depth):
            score = 5000
        
        # 5. Promotions
        elif move.promotion:
            if move.promotion == chess.QUEEN:
                score = 4500
            else:
                score = 4000
        
        # 6. Castling
        elif board.is_castling(move):
            score = 3500
        
        # 7. Basic positional bonus
        else:
            score = self._get_basic_positional_bonus(board, move)
        
        return score
    
    def _identify_tactical_features(self, board: chess.Board, move: chess.Move) -> List[str]:
        """Identify tactical patterns and themes in the move"""
        features = []
        
        # Make the move temporarily to analyze consequences
        board.push(move)
        
        try:
            # Check patterns
            if board.is_check():
                features.append("check")
                if board.is_checkmate():
                    features.append("checkmate")
            
            # Analyze the move's tactical implications
            from_square = move.from_square
            to_square = move.to_square
            moving_piece = board.piece_at(to_square)
            
            if moving_piece:
                # Fork detection
                if self._creates_fork(board, to_square, moving_piece):
                    features.append("fork")
                
                # Pin detection  
                if self._creates_pin(board, to_square, moving_piece):
                    features.append("pin")
                
                # Discovery detection
                if self._creates_discovery(board, from_square, to_square):
                    features.append("discovery")
                
                # Skewer detection
                if self._creates_skewer(board, to_square, moving_piece):
                    features.append("skewer")
            
            # Capture analysis
            board.pop()  # Undo to check capture
            if board.is_capture(move):
                features.append("capture")
                
                # Check if this breaks a pin
                if self._breaks_pin(board, move):
                    features.append("pin_break")
                
                # Check if this is a sacrifice
                if self._is_sacrifice(board, move):
                    features.append("sacrifice")
            
            board.push(move)  # Redo for other analysis
            
        finally:
            board.pop()  # Always restore position
        
        return features
    
    def _identify_positional_features(self, board: chess.Board, move: chess.Move) -> List[str]:
        """Identify positional characteristics of the move"""
        features = []
        
        moving_piece = board.piece_at(move.from_square)
        if not moving_piece:
            return features
        
        # Development features
        if self._is_development_move(board, move, moving_piece):
            features.append("development")
        
        # Central control
        center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        if move.to_square in center_squares:
            features.append("center_control")
        
        # King safety
        if self._improves_king_safety(board, move, moving_piece):
            features.append("king_safety")
        
        # Piece coordination
        if self._improves_piece_coordination(board, move, moving_piece):
            features.append("coordination")
        
        # Pawn structure
        if moving_piece.piece_type == chess.PAWN:
            if self._creates_pawn_chain(board, move):
                features.append("pawn_chain")
            if self._advances_passed_pawn(board, move):
                features.append("passed_pawn")
        
        # Space control
        if self._gains_space(board, move):
            features.append("space_gain")
        
        # Piece activity
        if self._increases_piece_activity(board, move, moving_piece):
            features.append("activity")
        
        return features
    
    def _determine_move_classes(self, board: chess.Board, move: chess.Move, 
                               tactical_features: List[str], positional_features: List[str]) -> Tuple[MoveClass, Optional[MoveClass]]:
        """Determine primary and secondary classifications for the move"""
        
        # Tactical moves take priority
        tactical_indicators = {"check", "checkmate", "fork", "pin", "discovery", "skewer", "sacrifice", "pin_break"}
        if any(feature in tactical_indicators for feature in tactical_features):
            return MoveClass.TACTICAL, MoveClass.OFFENSIVE
        
        # Captures and aggressive moves
        if board.is_capture(move) or "check" in tactical_features:
            return MoveClass.OFFENSIVE, None
        
        # Defensive moves
        if any(feature in {"king_safety", "pin_break"} for feature in tactical_features + positional_features):
            return MoveClass.DEFENSIVE, None
        
        # Development moves
        if "development" in positional_features or board.is_castling(move):
            return MoveClass.DEVELOPMENTAL, None
        
        # Endgame moves
        if self.current_position_tone and self.current_position_tone.endgame_factor > 0.7:
            if "activity" in positional_features or "passed_pawn" in positional_features:
                return MoveClass.ENDGAME, None
        
        # Default to developmental for quiet moves
        return MoveClass.DEVELOPMENTAL, None
    
    def _apply_dynamic_prioritization(self, classified_moves: List[ClassifiedMove]):
        """Apply dynamic prioritization based on position tone and search feedback"""
        if not self.current_position_tone:
            return
        
        tone = self.current_position_tone
        
        for classified_move in classified_moves:
            move_class = classified_move.primary_class
            bonus = 0
            
            # Apply position tone bonuses
            if move_class == MoveClass.OFFENSIVE:
                if tone.aggression_viability > 0.6:
                    bonus += 200
                if tone.tactical_complexity > 0.7:
                    bonus += 150
                # Apply search feedback
                bonus += int((tone.offensive_success_rate - 0.5) * 300)
            
            elif move_class == MoveClass.DEFENSIVE:
                if tone.king_safety_urgency > 0.6:
                    bonus += 250
                if tone.aggression_viability < 0.4:
                    bonus += 100
                # Apply search feedback
                bonus += int((tone.defensive_success_rate - 0.5) * 300)
            
            elif move_class == MoveClass.DEVELOPMENTAL:
                if tone.development_priority > 0.6:
                    bonus += 200
                if tone.endgame_factor < 0.3:
                    bonus += 100
                # Apply search feedback
                bonus += int((tone.developmental_success_rate - 0.5) * 300)
            
            elif move_class == MoveClass.TACTICAL:
                if tone.tactical_complexity > 0.5:
                    bonus += 300
                # Tactical moves always get priority if position supports them
            
            elif move_class == MoveClass.ENDGAME:
                if tone.endgame_factor > 0.7:
                    bonus += 200
            
            # Apply search feedback penalties for consistently failing move types
            if self.search_feedback.should_deprioritize_move_type(move_class):
                bonus -= 400
            
            classified_move.score += bonus
    
    # Helper methods for tactical pattern recognition
    def _creates_fork(self, board: chess.Board, square: int, piece: chess.Piece) -> bool:
        """Check if piece on square creates a fork"""
        if piece.piece_type not in {chess.KNIGHT, chess.PAWN, chess.KING}:
            return False
        
        attacks = board.attacks(square)
        enemy_pieces = []
        
        for attacked_square in attacks:
            attacked_piece = board.piece_at(attacked_square)
            if attacked_piece and attacked_piece.color != piece.color:
                if attacked_piece.piece_type in {chess.KING, chess.QUEEN, chess.ROOK}:
                    enemy_pieces.append(attacked_piece)
        
        return len(enemy_pieces) >= 2
    
    def _creates_pin(self, board: chess.Board, square: int, piece: chess.Piece) -> bool:
        """Check if piece on square creates a pin"""
        if piece.piece_type not in {chess.BISHOP, chess.ROOK, chess.QUEEN}:
            return False
        
        # This is a simplified pin detection - could be enhanced
        return False  # Placeholder for now
    
    def _creates_discovery(self, board: chess.Board, from_square: int, to_square: int) -> bool:
        """Check if move creates a discovered attack"""
        # Simplified discovery detection
        return False  # Placeholder for now
    
    def _creates_skewer(self, board: chess.Board, square: int, piece: chess.Piece) -> bool:
        """Check if piece on square creates a skewer"""
        # Simplified skewer detection
        return False  # Placeholder for now
    
    def _breaks_pin(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move breaks a pin"""
        # Simplified pin break detection
        return False  # Placeholder for now
    
    def _is_sacrifice(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if capture is a sacrifice"""
        if not board.is_capture(move):
            return False
        
        captured_piece = board.piece_at(move.to_square)
        moving_piece = board.piece_at(move.from_square)
        
        if not captured_piece or not moving_piece:
            return False
        
        captured_value = self.piece_values[captured_piece.piece_type]
        moving_value = self.piece_values[moving_piece.piece_type]
        
        return moving_value > captured_value + 100  # Sacrifice if losing >1 pawn
    
    # Helper methods for positional analysis
    def _is_development_move(self, board: chess.Board, move: chess.Move, piece: chess.Piece) -> bool:
        """Check if move develops a piece"""
        # Simplified: piece moving from back rank
        if piece.color == chess.WHITE:
            return chess.square_rank(move.from_square) <= 1 and chess.square_rank(move.to_square) > 1
        else:
            return chess.square_rank(move.from_square) >= 6 and chess.square_rank(move.to_square) < 6
    
    def _improves_king_safety(self, board: chess.Board, move: chess.Move, piece: chess.Piece) -> bool:
        """Check if move improves king safety"""
        return board.is_castling(move) or piece.piece_type == chess.KING
    
    def _improves_piece_coordination(self, board: chess.Board, move: chess.Move, piece: chess.Piece) -> bool:
        """Check if move improves piece coordination"""
        # Simplified: check if piece supports other pieces
        attacks_from_new_square = board.attacks(move.to_square)
        friendly_pieces = 0
        
        for square in attacks_from_new_square:
            attacked_piece = board.piece_at(square)
            if attacked_piece and attacked_piece.color == piece.color:
                friendly_pieces += 1
        
        return friendly_pieces >= 2
    
    def _creates_pawn_chain(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if pawn move creates or extends a pawn chain"""
        piece = board.piece_at(move.from_square)
        if not piece or piece.piece_type != chess.PAWN:
            return False
        
        # Check for adjacent pawns after move
        file = chess.square_file(move.to_square)
        rank = chess.square_rank(move.to_square)
        
        adjacent_squares = []
        if file > 0:
            adjacent_squares.append(chess.square(file - 1, rank))
        if file < 7:
            adjacent_squares.append(chess.square(file + 1, rank))
        
        for square in adjacent_squares:
            adj_piece = board.piece_at(square)
            if adj_piece and adj_piece.piece_type == chess.PAWN and adj_piece.color == piece.color:
                return True
        
        return False
    
    def _advances_passed_pawn(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move advances a passed pawn"""
        piece = board.piece_at(move.from_square)
        if not piece or piece.piece_type != chess.PAWN:
            return False
        
        # Simplified passed pawn detection
        file = chess.square_file(move.to_square)
        rank = chess.square_rank(move.to_square)
        
        # Check if any enemy pawns in adjacent files ahead
        enemy_color = not piece.color
        for check_file in [file - 1, file, file + 1]:
            if check_file < 0 or check_file > 7:
                continue
            
            for check_rank in range(rank + 1 if piece.color == chess.WHITE else 0, 
                                  8 if piece.color == chess.WHITE else rank):
                square = chess.square(check_file, check_rank)
                check_piece = board.piece_at(square)
                if check_piece and check_piece.piece_type == chess.PAWN and check_piece.color == enemy_color:
                    return False
        
        return True
    
    def _gains_space(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move gains space"""
        # Simplified: moving forward gains space
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        
        from_rank = chess.square_rank(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        
        if piece.color == chess.WHITE:
            return to_rank > from_rank
        else:
            return to_rank < from_rank
    
    def _increases_piece_activity(self, board: chess.Board, move: chess.Move, piece: chess.Piece) -> bool:
        """Check if move increases piece activity"""
        # Count squares attacked from new position vs old position
        old_attacks = len(board.attacks(move.from_square))
        new_attacks = len(board.attacks(move.to_square))
        
        return new_attacks > old_attacks
    
    # Position analysis helper methods
    def _calculate_material(self, board: chess.Board, color: bool) -> int:
        """Calculate total material for a color"""
        material = 0
        for piece_type, value in self.piece_values.items():
            if piece_type != chess.KING:
                material += len(board.pieces(piece_type, color)) * value
        return material
    
    def _assess_tactical_complexity(self, board: chess.Board) -> float:
        """Assess how tactically complex the position is"""
        complexity = 0.0
        
        # More pieces on board = more tactical
        total_pieces = len(board.piece_map())
        complexity += min(total_pieces / 32.0, 1.0) * 0.3
        
        # Checks increase tactical complexity
        if board.is_check():
            complexity += 0.4
        
        # Captures available
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        complexity += min(len(captures) / 10.0, 1.0) * 0.3
        
        return min(complexity, 1.0)
    
    def _assess_king_safety(self, board: chess.Board) -> float:
        """Assess king safety urgency"""
        urgency = 0.0
        
        if board.is_check():
            urgency += 0.6
        
        # Count attackers near king
        our_color = board.turn
        king_square = board.king(our_color)
        
        if king_square is not None:
            king_zone = []
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            
            for file_offset in [-1, 0, 1]:
                for rank_offset in [-1, 0, 1]:
                    new_file = king_file + file_offset
                    new_rank = king_rank + rank_offset
                    if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
                        king_zone.append(chess.square(new_file, new_rank))
            
            attackers = 0
            for square in king_zone:
                if board.is_attacked_by(not our_color, square):
                    attackers += 1
            
            urgency += min(attackers / 8.0, 0.4)
        
        return min(urgency, 1.0)
    
    def _assess_development_needs(self, board: chess.Board) -> float:
        """Assess how much development is still needed"""
        our_color = board.turn
        development_score = 0.0
        
        # Check minor pieces on back rank
        back_rank = 0 if our_color == chess.WHITE else 7
        
        for file in range(8):
            square = chess.square(file, back_rank)
            piece = board.piece_at(square)
            if piece and piece.color == our_color and piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
                development_score += 0.25
        
        # Check if king has castled
        if our_color == chess.WHITE:
            if not (board.has_castling_rights(chess.WHITE) or 
                   board.king(chess.WHITE) in {chess.G1, chess.C1}):
                development_score += 0.3
        else:
            if not (board.has_castling_rights(chess.BLACK) or 
                   board.king(chess.BLACK) in {chess.G8, chess.C8}):
                development_score += 0.3
        
        return min(development_score, 1.0)
    
    def _assess_endgame_factor(self, board: chess.Board, total_material: int) -> float:
        """Assess how much we're in the endgame"""
        # Opening material is roughly 7800 (without kings)
        opening_material = 7800
        endgame_threshold = 2000
        
        if total_material <= endgame_threshold:
            return 1.0
        elif total_material >= opening_material:
            return 0.0
        else:
            return 1.0 - ((total_material - endgame_threshold) / (opening_material - endgame_threshold))
    
    def _assess_aggression_viability(self, board: chess.Board) -> float:
        """Assess whether aggressive moves are likely to succeed"""
        viability = 0.5  # Start neutral
        
        our_color = board.turn
        enemy_king = board.king(not our_color)
        
        if enemy_king is not None:
            # Enemy king safety affects our aggression viability
            enemy_safety = self._assess_king_safety_for_color(board, not our_color)
            viability += (1.0 - enemy_safety) * 0.4
        
        # Our material advantage affects aggression viability
        our_material = self._calculate_material(board, our_color)
        enemy_material = self._calculate_material(board, not our_color)
        
        if our_material > enemy_material:
            advantage = (our_material - enemy_material) / max(enemy_material, 1)
            viability += min(advantage, 0.3)
        
        return min(max(viability, 0.0), 1.0)
    
    def _assess_king_safety_for_color(self, board: chess.Board, color: bool) -> float:
        """Assess king safety for specific color"""
        king_square = board.king(color)
        if king_square is None:
            return 0.0
        
        safety = 1.0
        
        # Check for attacks around king
        king_zone = []
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        for file_offset in [-1, 0, 1]:
            for rank_offset in [-1, 0, 1]:
                new_file = king_file + file_offset
                new_rank = king_rank + rank_offset
                if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
                    king_zone.append(chess.square(new_file, new_rank))
        
        attackers = 0
        for square in king_zone:
            if board.is_attacked_by(not color, square):
                attackers += 1
        
        safety -= min(attackers / 8.0, 0.8)
        
        return max(safety, 0.0)
    
    # Traditional move ordering support methods
    def _is_killer_move(self, move: chess.Move, depth: int) -> bool:
        """Check if move is a killer move at this depth"""
        if depth not in self.killer_moves:
            return False
        return move in self.killer_moves[depth]
    
    def store_killer_move(self, move: chess.Move, depth: int):
        """Store a killer move"""
        if depth not in self.killer_moves:
            self.killer_moves[depth] = []
        
        if move not in self.killer_moves[depth]:
            self.killer_moves[depth].insert(0, move)
            if len(self.killer_moves[depth]) > 2:
                self.killer_moves[depth].pop()
    
    def _get_basic_positional_bonus(self, board: chess.Board, move: chess.Move) -> int:
        """Basic positional bonus for quiet moves"""
        bonus = 100
        
        piece = board.piece_at(move.from_square)
        if not piece:
            return bonus
        
        # Center control
        center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        if move.to_square in center_squares:
            bonus += 50
        
        # Piece-specific bonuses
        if piece.piece_type == chess.KNIGHT:
            if move.to_square in {chess.C3, chess.F3, chess.C6, chess.F6}:
                bonus += 30
        elif piece.piece_type == chess.BISHOP:
            if move.to_square in {chess.C4, chess.F4, chess.C5, chess.F5}:
                bonus += 25
        
        return bonus
    
    def record_search_result(self, move: chess.Move, move_class: MoveClass, 
                           evaluation: float, improved: bool):
        """Record the result of trying a move during search"""
        self.search_feedback.record_move_result(move_class, evaluation, improved)
    
    def reset_for_new_position(self):
        """Reset dynamic state for new position"""
        self.current_position_tone = None
        self.search_feedback = SearchFeedback()
    
    def clear_killers(self):
        """Clear all killer moves (for new game)"""
        self.killer_moves.clear()
        self.reset_for_new_position()