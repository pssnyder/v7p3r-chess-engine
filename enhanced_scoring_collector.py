#!/usr/bin/env python3
"""
Enhanced Scoring Data Collector for V7P3R Chess Engine
This module extracts detailed scoring breakdown from the v7p3r scoring system
"""

import chess
import json
from typing import Dict, Any, Optional
import logging

class EnhancedScoringCollector:
    """
    Collects detailed scoring breakdown from v7p3r engine evaluations
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
        self.scoring_components = [
            'checkmate_threats_score',
            'king_safety_score', 
            'king_threat_score',
            'king_endangerment_score',
            'draw_scenarios_score',
            'material_score',
            'pst_score',
            'piece_coordination_score',
            'center_control_score',
            'pawn_structure_score',
            'pawn_weaknesses_score',
            'passed_pawns_score',
            'pawn_majority_score',
            'bishop_pair_score',
            'knight_pair_score',
            'bishop_vision_score',
            'rook_coordination_score',
            'castling_protection_score',
            'castling_sacrifice_score',
            'piece_activity_score',
            'improved_minor_piece_activity_score',
            'mobility_score',
            'undeveloped_pieces_score',
            'hanging_pieces_score',
            'undefended_pieces_score',
            'queen_capture_score',
            'tempo_bonus_score',
            'en_passant_score',
            'open_files_score',
            'stalemate_score'
        ]
    
    def collect_detailed_scoring(self, scoring_calculator, board: chess.Board, color: chess.Color) -> Dict[str, float]:
        """
        Collect detailed scoring breakdown by calling individual scoring methods
        """
        detailed_scores = {}
        
        try:
            # Get the overall score first (this already exists)
            total_score = scoring_calculator.calculate_score(board, color)
            detailed_scores['total_score'] = total_score
            
            # Now collect individual component scores
            detailed_scores['checkmate_threats_score'] = self._safe_call(
                scoring_calculator._checkmate_threats, board, color)
            detailed_scores['king_safety_score'] = self._safe_call(
                scoring_calculator._king_safety, board, color)
            detailed_scores['king_threat_score'] = self._safe_call(
                scoring_calculator._king_threat, board, color)
            detailed_scores['king_endangerment_score'] = self._safe_call(
                scoring_calculator._king_endangerment, board, color)
            detailed_scores['draw_scenarios_score'] = self._safe_call(
                scoring_calculator._draw_scenarios, board)
            detailed_scores['material_score'] = self._safe_call(
                scoring_calculator._material_score, board, color)
            
            # PST score requires special handling
            if hasattr(scoring_calculator, 'pst'):
                pst_score = scoring_calculator.pst.evaluate_board_position(board, 0.0)
                if color == chess.BLACK:
                    pst_score = -pst_score
                detailed_scores['pst_score'] = pst_score
            else:
                detailed_scores['pst_score'] = 0.0
            
            detailed_scores['piece_coordination_score'] = self._safe_call(
                scoring_calculator._piece_coordination, board, color)
            detailed_scores['center_control_score'] = self._safe_call(
                scoring_calculator._center_control, board, color)
            detailed_scores['pawn_structure_score'] = self._safe_call(
                scoring_calculator._pawn_structure, board, color)
            detailed_scores['pawn_weaknesses_score'] = self._safe_call(
                scoring_calculator._pawn_weaknesses, board, color)
            detailed_scores['passed_pawns_score'] = self._safe_call(
                scoring_calculator._passed_pawns, board, color)
            detailed_scores['pawn_majority_score'] = self._safe_call(
                scoring_calculator._pawn_majority, board, color)
            detailed_scores['bishop_pair_score'] = self._safe_call(
                scoring_calculator._bishop_pair, board, color)
            detailed_scores['knight_pair_score'] = self._safe_call(
                scoring_calculator._knight_pair, board, color)
            detailed_scores['bishop_vision_score'] = self._safe_call(
                scoring_calculator._bishop_vision, board, color)
            detailed_scores['rook_coordination_score'] = self._safe_call(
                scoring_calculator._rook_coordination, board, color)
            detailed_scores['castling_protection_score'] = self._safe_call(
                scoring_calculator._castling_protection, board, color)
            detailed_scores['castling_sacrifice_score'] = self._safe_call(
                scoring_calculator._castling_sacrifice, board, color)
            detailed_scores['piece_activity_score'] = self._safe_call(
                scoring_calculator._piece_activity, board, color)
            detailed_scores['improved_minor_piece_activity_score'] = self._safe_call(
                scoring_calculator._improved_minor_piece_activity, board, color)
            detailed_scores['mobility_score'] = self._safe_call(
                scoring_calculator._mobility_score, board, color)
            detailed_scores['undeveloped_pieces_score'] = self._safe_call(
                scoring_calculator._undeveloped_pieces, board, color)
            detailed_scores['hanging_pieces_score'] = self._safe_call(
                scoring_calculator._hanging_pieces, board, color)
            detailed_scores['undefended_pieces_score'] = self._safe_call(
                scoring_calculator._undefended_pieces, board, color)
            detailed_scores['queen_capture_score'] = self._safe_call(
                scoring_calculator._queen_capture, board, color)
            detailed_scores['tempo_bonus_score'] = self._safe_call(
                scoring_calculator._tempo_bonus, board, color)
            detailed_scores['en_passant_score'] = self._safe_call(
                scoring_calculator._en_passant, board, color)
            detailed_scores['open_files_score'] = self._safe_call(
                scoring_calculator._open_files, board, color)
            detailed_scores['stalemate_score'] = self._safe_call(
                scoring_calculator._stalemate, board)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error collecting detailed scoring: {e}")
            # Return zeros for all components if collection fails
            for component in self.scoring_components:
                if component not in detailed_scores:
                    detailed_scores[component] = 0.0
            detailed_scores['total_score'] = 0.0
        
        return detailed_scores
    
    def _safe_call(self, method, *args) -> float:
        """
        Safely call a scoring method and return 0.0 if it fails
        """
        try:
            result = method(*args)
            return float(result) if result is not None else 0.0
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error calling {method.__name__}: {e}")
            return 0.0
    
    def analyze_game_phase(self, board: chess.Board) -> str:
        """
        Determine the current game phase based on material and piece count
        """
        try:
            # Count major pieces (excluding kings and pawns)
            major_pieces = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    major_pieces += 1
            
            # Simple game phase classification
            if major_pieces >= 14:  # Most pieces still on board
                return 'opening'
            elif major_pieces >= 8:  # Some pieces traded
                return 'middlegame'
            else:  # Few pieces remaining
                return 'endgame'
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error analyzing game phase: {e}")
            return 'unknown'
    
    def classify_position_type(self, board: chess.Board) -> str:
        """
        Classify the position as tactical, positional, or balanced
        """
        try:
            # Simple classification based on material balance and piece activity
            # This is a basic implementation - could be enhanced with more sophisticated analysis
            
            # Count captures available
            captures = 0
            for move in board.legal_moves:
                if board.is_capture(move):
                    captures += 1
            
            # Count checks available
            checks = 0
            for move in board.legal_moves:
                board.push(move)
                if board.is_check():
                    checks += 1
                board.pop()
            
            # Basic classification
            if captures >= 3 or checks >= 2:
                return 'tactical'
            elif captures == 0 and checks == 0:
                return 'positional'
            else:
                return 'balanced'
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error classifying position type: {e}")
            return 'unknown'
    
    def calculate_material_balance(self, board: chess.Board) -> float:
        """
        Calculate material balance (positive favors white)
        """
        try:
            piece_values = {
                chess.PAWN: 1.0,
                chess.KNIGHT: 3.0,
                chess.BISHOP: 3.25,
                chess.ROOK: 5.0,
                chess.QUEEN: 9.0,
                chess.KING: 0.0
            }
            
            white_material = 0.0
            black_material = 0.0
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    value = piece_values.get(piece.piece_type, 0.0)
                    if piece.color == chess.WHITE:
                        white_material += value
                    else:
                        black_material += value
            
            return white_material - black_material
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calculating material balance: {e}")
            return 0.0
    
    def count_pieces(self, board: chess.Board) -> int:
        """
        Count total pieces on the board
        """
        try:
            return len(board.piece_map())
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error counting pieces: {e}")
            return 0
    
    def classify_move_type(self, board: chess.Board, move: chess.Move) -> str:
        """
        Classify the type of move being made
        """
        try:
            if board.is_castling(move):
                return 'castling'
            elif board.is_capture(move):
                return 'capture'
            elif board.is_en_passant(move):
                return 'en_passant'
            elif move.promotion:
                return 'promotion'
            else:
                return 'normal'
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error classifying move type: {e}")
            return 'unknown'
    
    def calculate_search_efficiency_metrics(self, nodes_searched: int, time_taken: float, depth: int) -> Dict[str, float]:
        """
        Calculate search efficiency metrics
        """
        metrics = {}
        
        try:
            # Nodes per second
            if time_taken > 0:
                metrics['nps'] = nodes_searched / time_taken
            else:
                metrics['nps'] = 0.0
            
            # Effective branching factor (approximate)
            if depth > 0 and nodes_searched > 0:
                metrics['branching_factor'] = pow(nodes_searched, 1.0/depth)
            else:
                metrics['branching_factor'] = 0.0
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calculating search efficiency: {e}")
            metrics['nps'] = 0.0
            metrics['branching_factor'] = 0.0
        
        return metrics
