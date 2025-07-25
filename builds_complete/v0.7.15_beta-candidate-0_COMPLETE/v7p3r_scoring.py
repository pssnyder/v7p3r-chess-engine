# v7p3r_scoring.py

"""Main Scoring Module for V7P3R Chess Engine
Coordinates all scoring components and provides unified evaluation interface.
"""

import chess
from v7p3r_tempo import TempoCalculation
from v7p3r_primary_scoring import PrimaryScoring
from v7p3r_secondary_scoring import SecondaryScoring
from v7p3r_quiescence import QuiescenceSearch

class ScoringSystem:
    def __init__(self, config):
        self.config = config
        self.tempo = TempoCalculation()
        self.primary = PrimaryScoring()
        self.secondary = SecondaryScoring(config)
        self.quiescence = QuiescenceSearch()
        
        # Configuration flags
        self.use_tempo = config.is_enabled('engine_config', 'use_tempo_scoring')
        self.use_primary = config.is_enabled('engine_config', 'use_primary_scoring')
        self.use_secondary = config.is_enabled('engine_config', 'use_secondary_scoring')
        self.use_quiescence = config.is_enabled('engine_config', 'use_quiescence')
    
    def evaluate_move(self, board, move, our_color, depth=0, alpha=None, beta=None):
        """Comprehensive move evaluation"""
        total_score = 0
        evaluation_details = {}
        critical_move = False
        
        # Make the move to evaluate resulting position
        board_copy = board.copy()
        board_copy.push(move)
        
        # 1. Tempo Calculation (critical moves)
        if self.use_tempo:
            tempo_score, is_critical = self.tempo.evaluate_tempo(board, move, depth)
            total_score += tempo_score
            evaluation_details['tempo'] = tempo_score
            critical_move = is_critical
            
            # Short circuit for critical moves
            if self.tempo.should_short_circuit(tempo_score):
                evaluation_details['short_circuit'] = True
                return total_score, evaluation_details, critical_move
        
        # 2. Primary Scoring
        if self.use_primary:
            primary_eval = self.primary.evaluate_primary_score(board_copy, our_color)
            total_score += primary_eval['total']
            evaluation_details['primary'] = primary_eval
        
        # 3. Secondary Scoring
        if self.use_secondary:
            material_score = evaluation_details.get('primary', {}).get('material_score', 0)
            secondary_eval = self.secondary.evaluate_secondary_score(board, move, our_color, material_score)
            total_score += secondary_eval['total']
            evaluation_details['secondary'] = secondary_eval
        
        # 4. Quiescence Search (if position is not quiet)
        if self.use_quiescence and not self.quiescence.is_quiet_position(board_copy):
            if alpha is not None and beta is not None:
                quies_score = self.quiescence.quiescence_search(
                    board_copy, alpha, beta, our_color, self.primary
                )
                # Adjust total score based on quiescence result
                total_score += (quies_score - evaluation_details.get('primary', {}).get('total', 0)) // 2
                evaluation_details['quiescence'] = quies_score
        
        return total_score, evaluation_details, critical_move
    
    def evaluate_position(self, board, our_color):
        """Static position evaluation without move"""
        if self.use_primary:
            primary_eval = self.primary.evaluate_primary_score(board, our_color)
            return primary_eval['total']
        return 0
    
    def get_material_balance(self, board, our_color):
        """Get current material balance"""
        return self.primary.get_material_balance(board, our_color)
