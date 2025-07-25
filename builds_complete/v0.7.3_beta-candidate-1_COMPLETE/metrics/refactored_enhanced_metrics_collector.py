#!/usr/bin/env python3
"""
Refactored Enhanced Metrics Collector for V7P3R Chess Engine
Now uses search_dataset and score_dataset as the canonical source for metrics collection
This ensures all relevant data is captured consistently for each move
"""

import chess
import json
from typing import Dict, Any, Optional
import logging

class RefactoredEnhancedMetricsCollector:
    """
    Collects comprehensive metrics from v7p3r engine using the new dataset objects
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
        
    def collect_from_search_dataset(self, search_engine) -> Dict[str, Any]:
        """
        Extract metrics from the search_dataset object in v7p3r_search
        """
        search_metrics = {}
        
        try:
            if hasattr(search_engine, 'search_dataset') and isinstance(search_engine.search_dataset, dict):
                dataset = search_engine.search_dataset
                
                # Core search metrics
                search_metrics.update({
                    'search_id': dataset.get('search_id', ''),
                    'search_algorithm': dataset.get('search_algorithm', 'unknown'),
                    'depth_reached': dataset.get('depth', 0),
                    'max_depth': dataset.get('max_depth', 0),
                    'nodes_searched': dataset.get('nodes_searched', 0),
                    'evaluation': dataset.get('evaluation', 0.0),
                    'best_score': dataset.get('best_score', 0.0),
                    'color_name': dataset.get('color_name', 'unknown'),
                    'fen': dataset.get('fen', ''),
                    'root_board_fen': dataset.get('root_board_fen', ''),
                })
                
                # Convert best_move from chess.Move to string
                best_move = dataset.get('best_move')
                if best_move and best_move != chess.Move.null():
                    search_metrics['best_move'] = str(best_move)
                else:
                    search_metrics['best_move'] = None
                
                # Extract principal variation
                pv_stack = dataset.get('pv_move_stack', [])
                if pv_stack:
                    search_metrics['principal_variation_length'] = len(pv_stack)
                    # Convert PV moves to string format
                    pv_moves = []
                    for pv_dict in pv_stack:
                        if isinstance(pv_dict, dict):
                            for move in pv_dict.values():
                                if move and move != chess.Move.null():
                                    pv_moves.append(str(move))
                    search_metrics['principal_variation'] = ' '.join(pv_moves[:10])  # Limit PV length
                else:
                    search_metrics['principal_variation_length'] = 0
                    search_metrics['principal_variation'] = ''
                
                if self.logger:
                    self.logger.debug(f"Extracted {len(search_metrics)} search metrics from search_dataset")
                    
            else:
                if self.logger:
                    self.logger.warning("No search_dataset found in search engine")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error extracting search dataset: {e}")
                
        return search_metrics
    
    def collect_from_score_dataset(self, scoring_calculator) -> Dict[str, Any]:
        """
        Extract detailed scoring metrics from the score_dataset object in v7p3r_score
        """
        scoring_metrics = {}
        
        try:
            if hasattr(scoring_calculator, 'score_dataset') and isinstance(scoring_calculator.score_dataset, dict):
                dataset = scoring_calculator.score_dataset
                
                # Position information
                scoring_metrics.update({
                    'fen_before': dataset.get('fen', ''),
                    'color': dataset.get('color', 'unknown'),
                    'current_player': dataset.get('current_player', ''),
                    'v7p3r_thinking': dataset.get('v7p3r_thinking', False),
                })
                
                # Convert move from chess.Move to string
                move = dataset.get('move')
                if move and move != chess.Move.null():
                    scoring_metrics['last_move'] = str(move)
                else:
                    scoring_metrics['last_move'] = None
                
                # Piece information
                piece_type = dataset.get('piece')
                if piece_type:
                    scoring_metrics['piece_type'] = chess.piece_name(piece_type) if piece_type else None
                else:
                    scoring_metrics['piece_type'] = None
                
                # Extract all individual scoring components
                scoring_components = [
                    'checkmate_threats', 'king_safety', 'king_attack', 'draw_scenarios',
                    'material_score', 'pst_score', 'piece_coordination', 'center_control',
                    'pawn_structure', 'passed_pawns', 'pawn_count', 'pawn_promotion',
                    'bishop_count', 'knight_count', 'bishop_vision', 'rook_coordination',
                    'castling_protection', 'piece_activity', 'bishop_activity', 'mobility',
                    'piece_development', 'piece_attacks', 'piece_protection', 'queen_attack',
                    'piece_captures', 'tempo', 'en_passant', 'open_files'
                ]
                
                # Extract component scores with proper prefixing
                for component in scoring_components:
                    value = dataset.get(component, 0.0)
                    scoring_metrics[f'{component}_score'] = float(value) if value is not None else 0.0
                
                # Calculate total score from components (if not already present)
                if 'total_score' not in dataset:
                    total = sum(scoring_metrics[f'{comp}_score'] for comp in scoring_components 
                              if f'{comp}_score' in scoring_metrics)
                    scoring_metrics['total_score'] = total
                else:
                    scoring_metrics['total_score'] = float(dataset.get('total_score', 0.0))
                
                if self.logger:
                    self.logger.debug(f"Extracted {len(scoring_metrics)} scoring metrics from score_dataset")
                    
            else:
                if self.logger:
                    self.logger.warning("No score_dataset found in scoring calculator")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error extracting score dataset: {e}")
                
        return scoring_metrics
    
    def collect_legacy_scoring_dictionary(self, scoring_calculator) -> Dict[str, Any]:
        """
        Extract the legacy scoring dictionary as a fallback
        """
        scoring_dict = {}
        
        try:
            if hasattr(scoring_calculator, 'scoring') and isinstance(scoring_calculator.scoring, dict):
                # Copy the scoring dictionary with proper type conversion
                for key, value in scoring_calculator.scoring.items():
                    if key == 'move':
                        # Convert chess.Move to string representation
                        if value and value != chess.Move.null():
                            scoring_dict[f'legacy_{key}'] = str(value)
                        else:
                            scoring_dict[f'legacy_{key}'] = None
                    elif key == 'fen':
                        scoring_dict[f'legacy_{key}'] = str(value) if value else None
                    elif isinstance(value, (int, float)):
                        scoring_dict[f'legacy_{key}'] = float(value)
                    else:
                        scoring_dict[f'legacy_{key}'] = str(value) if value is not None else None
                        
                if self.logger:
                    self.logger.debug(f"Extracted legacy scoring dictionary with {len(scoring_dict)} components")
            else:
                if self.logger:
                    self.logger.debug("No legacy scoring dictionary found")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error extracting legacy scoring dictionary: {e}")
                
        return scoring_dict
    
    def collect_comprehensive_metrics(self, v7p3r_engine, board: chess.Board, move: chess.Move, 
                                    time_taken: float = 0.0) -> Dict[str, Any]:
        """
        Collect all available metrics from the v7p3r engine dataset objects
        """
        comprehensive_metrics = {}
        
        try:
            # Extract search metrics
            if hasattr(v7p3r_engine, 'search_engine'):
                search_metrics = self.collect_from_search_dataset(v7p3r_engine.search_engine)
                comprehensive_metrics.update(search_metrics)
                
                # Calculate search efficiency if we have the data
                nodes = search_metrics.get('nodes_searched', 0)
                depth = search_metrics.get('depth_reached', 0)
                if nodes > 0 and time_taken > 0:
                    comprehensive_metrics['nps'] = nodes / time_taken
                    comprehensive_metrics['time_taken'] = time_taken
                    comprehensive_metrics['time_efficiency'] = time_taken / max(1.0, time_taken)
                    
                    # Effective branching factor
                    if depth > 0:
                        comprehensive_metrics['branching_factor'] = pow(nodes, 1.0/depth)
                    else:
                        comprehensive_metrics['branching_factor'] = 0.0
                else:
                    comprehensive_metrics.update({
                        'nps': 0.0,
                        'time_taken': time_taken,
                        'time_efficiency': 0.0,
                        'branching_factor': 0.0
                    })
            
            # Extract scoring metrics
            if hasattr(v7p3r_engine, 'scoring_calculator'):
                scoring_metrics = self.collect_from_score_dataset(v7p3r_engine.scoring_calculator)
                comprehensive_metrics.update(scoring_metrics)
                
                # Also get legacy scoring as backup
                legacy_metrics = self.collect_legacy_scoring_dictionary(v7p3r_engine.scoring_calculator)
                comprehensive_metrics.update(legacy_metrics)
            
            # Add position analysis
            position_metrics = self.analyze_position(board, move)
            comprehensive_metrics.update(position_metrics)
            
            # Add engine configuration
            engine_metrics = self.collect_engine_configuration(v7p3r_engine)
            comprehensive_metrics.update(engine_metrics)
            
            if self.logger:
                self.logger.info(f"Collected comprehensive metrics with {len(comprehensive_metrics)} components")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error collecting comprehensive metrics: {e}")
                
        return comprehensive_metrics
    
    def analyze_position(self, board: chess.Board, move: chess.Move) -> Dict[str, Any]:
        """
        Analyze the current position for game phase, material balance, etc.
        """
        analysis = {}
        
        try:
            # Game phase analysis
            piece_count = len(board.piece_map())
            analysis['piece_count'] = piece_count
            
            if piece_count >= 28:
                analysis['game_phase'] = 'opening'
            elif piece_count >= 16:
                analysis['game_phase'] = 'middlegame'
            else:
                analysis['game_phase'] = 'endgame'
            
            # Material balance calculation
            piece_values = {
                chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.25,
                chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0
            }
            
            white_material = black_material = 0.0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    value = piece_values.get(piece.piece_type, 0.0)
                    if piece.color == chess.WHITE:
                        white_material += value
                    else:
                        black_material += value
            
            analysis['material_balance'] = white_material - black_material
            analysis['white_material'] = white_material
            analysis['black_material'] = black_material
            
            # Move classification
            analysis['is_capture'] = board.is_capture(move)
            analysis['is_check'] = board.gives_check(move)
            analysis['is_castling'] = board.is_castling(move)
            analysis['is_en_passant'] = board.is_en_passant(move)
            analysis['is_promotion'] = bool(move.promotion)
            
            # Position type classification
            captures_available = sum(1 for m in board.legal_moves if board.is_capture(m))
            checks_available = sum(1 for m in board.legal_moves if board.gives_check(m))
            
            if captures_available >= 3 or checks_available >= 2:
                analysis['position_type'] = 'tactical'
            elif captures_available == 0 and checks_available == 0:
                analysis['position_type'] = 'positional'
            else:
                analysis['position_type'] = 'balanced'
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error analyzing position: {e}")
            
        return analysis
    
    def collect_engine_configuration(self, v7p3r_engine) -> Dict[str, Any]:
        """
        Collect engine configuration information
        """
        config_metrics = {}
        
        try:
            config_metrics.update({
                'engine_name': getattr(v7p3r_engine, 'name', 'v7p3r'),
                'engine_version': getattr(v7p3r_engine, 'version', 'unknown'),
                'algorithm': getattr(v7p3r_engine, 'search_algorithm', 'minimax'),
                'default_depth': getattr(v7p3r_engine, 'depth', 3),
                'max_depth': getattr(v7p3r_engine, 'max_depth', 5),
            })
            
            # Get ruleset information from scoring calculator
            if hasattr(v7p3r_engine, 'scoring_calculator'):
                scorer = v7p3r_engine.scoring_calculator
                config_metrics.update({
                    'ruleset_name': getattr(scorer, 'ruleset_name', 'unknown'),
                    'monitoring_enabled': getattr(scorer, 'monitoring_enabled', False),
                    'verbose_output': getattr(scorer, 'verbose_output_enabled', False),
                })
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error collecting engine configuration: {e}")
                
        return config_metrics

    def validate_metrics_completeness(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that all essential metrics are present and add defaults for missing ones
        """
        essential_metrics = {
            'search_algorithm': 'unknown',
            'depth_reached': 0,
            'nodes_searched': 0,
            'evaluation': 0.0,
            'total_score': 0.0,
            'game_phase': 'unknown',
            'material_balance': 0.0,
            'nps': 0.0,
            'branching_factor': 0.0,
            'position_type': 'unknown',
            'piece_count': 0,
            'engine_name': 'v7p3r'
        }
        
        for key, default_value in essential_metrics.items():
            if key not in metrics:
                metrics[key] = default_value
                if self.logger:
                    self.logger.debug(f"Added missing metric '{key}' with default value: {default_value}")
        
        return metrics

if __name__ == "__main__":
    """Test the refactored enhanced metrics collector"""
    import sys
    import os
    
    # Add the v7p3r_engine path to import modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v7p3r_engine'))
    
    try:
        from v7p3r_engine.v7p3r import v7p3rEngine
        
        print("Testing Refactored Enhanced Metrics Collector")
        print("=" * 50)
        
        # Create test board and engine
        board = chess.Board()
        engine = v7p3rEngine()
        
        # Create refactored collector
        collector = RefactoredEnhancedMetricsCollector()
        
        # Make a test move
        legal_moves = list(board.legal_moves)
        if legal_moves:
            test_move = legal_moves[0]
            
            # Collect comprehensive metrics
            metrics = collector.collect_comprehensive_metrics(
                engine, board, test_move, time_taken=0.1
            )
            
            print(f"Collected {len(metrics)} total metrics:")
            print("\nSearch Metrics:")
            search_keys = [k for k in metrics.keys() if any(x in k for x in ['search', 'depth', 'nodes', 'evaluation', 'nps', 'branching'])]
            for key in search_keys:
                print(f"  {key}: {metrics[key]}")
            
            print("\nScoring Metrics:")
            scoring_keys = [k for k in metrics.keys() if '_score' in k]
            non_zero_scores = {k: v for k, v in metrics.items() if k in scoring_keys and v != 0.0}
            print(f"  Non-zero scores: {len(non_zero_scores)} / {len(scoring_keys)}")
            for key, value in list(non_zero_scores.items())[:10]:  # Show first 10
                print(f"    {key}: {value}")
            
            print("\nPosition Analysis:")
            position_keys = [k for k in metrics.keys() if any(x in k for x in ['game_phase', 'material', 'position_type', 'piece_count'])]
            for key in position_keys:
                print(f"  {key}: {metrics[key]}")
            
            print("\nValidation:")
            validated_metrics = collector.validate_metrics_completeness(metrics)
            print(f"  Metrics after validation: {len(validated_metrics)}")
            
        else:
            print("No legal moves available for testing")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running this from the correct directory")
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
