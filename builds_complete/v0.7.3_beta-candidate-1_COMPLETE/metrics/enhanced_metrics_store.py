#!/usr/bin/env python3
"""
Enhanced Metrics Store for V7P3R Chess Engine
Handles storage of comprehensive chess engine metrics with detailed scoring breakdown
"""

import sqlite3
import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List  # Ensure Optional is imported
import logging

class EnhancedMetricsStore:
    """
    Enhanced metrics storage system with comprehensive data collection
    """
    
    def __init__(self, db_path: str = "metrics/chess_metrics_v2.db", logger: Optional[logging.Logger] = None):
        self.db_path = db_path
        self.logger = logger
        self._ensure_db_directory()
        self._initialize_database()
    
    def _ensure_db_directory(self):
        """Ensure the metrics directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _initialize_database(self):
        """Initialize the database with the enhanced schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Read and execute the enhanced schema
            try:
                from enhanced_schema import ENHANCED_CHESS_METRICS_SCHEMA
            except ImportError:
                # Try absolute import
                from metrics.enhanced_schema import ENHANCED_CHESS_METRICS_SCHEMA
            cursor.executescript(ENHANCED_CHESS_METRICS_SCHEMA)
            
            conn.commit()
            conn.close()
            
            if self.logger:
                self.logger.info(f"Enhanced metrics database initialized at {self.db_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error initializing database: {e}")
            raise
    
    def add_engine_config(self, engine_config: Dict[str, Any]) -> str:
        """
        Add or retrieve engine configuration
        Returns the config_id for the configuration
        """
        try:
            # Create a hash of the configuration for unique identification
            config_str = json.dumps(engine_config, sort_keys=True)
            config_id = hashlib.md5(config_str.encode()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if config already exists
            cursor.execute("SELECT config_id FROM engine_configs WHERE config_id = ?", (config_id,))
            if cursor.fetchone():
                conn.close()
                return config_id
            
            # Insert new configuration
            cursor.execute("""
                INSERT INTO engine_configs (
                    config_id, engine_name, engine_version, search_algorithm,
                    depth, max_depth, max_moves, ruleset, use_game_phase,
                    time_control, other_settings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config_id,
                engine_config.get('name', 'unknown'),
                engine_config.get('version', 'unknown'),
                engine_config.get('search_algorithm', 'unknown'),
                engine_config.get('depth', 0),
                engine_config.get('max_depth', 0),
                engine_config.get('max_moves', 0),
                engine_config.get('ruleset', 'unknown'),
                engine_config.get('use_game_phase', False),
                json.dumps(engine_config.get('time_control', {})),
                json.dumps({k: v for k, v in engine_config.items() 
                           if k not in ['name', 'version', 'search_algorithm', 'depth', 
                                      'max_depth', 'max_moves', 'ruleset', 'use_game_phase', 'time_control']})
            ))
            
            conn.commit()
            conn.close()
            
            if self.logger:
                self.logger.info(f"Added engine configuration: {config_id}")
            
            return config_id
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error adding engine config: {e}")
            raise
    
    def start_game(self, game_id: str, white_player: str, black_player: str, 
                   white_config: Dict[str, Any], black_config: Dict[str, Any],
                   pgn_filename: Optional[str] = None) -> None:
        """
        Register a new game with engine configurations
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Add engine configurations
            white_config_id = self.add_engine_config(white_config)
            black_config_id = self.add_engine_config(black_config)
            
            # Insert game record
            cursor.execute("""
                INSERT OR REPLACE INTO games (
                    game_id, pgn_filename, timestamp, white_player, black_player
                ) VALUES (?, ?, ?, ?, ?)
            """, (game_id, pgn_filename, datetime.now().isoformat(), white_player, black_player))
            
            # Link game to engine configurations
            cursor.execute("""
                INSERT OR REPLACE INTO game_participants (game_id, player_color, config_id)
                VALUES (?, 'white', ?)
            """, (game_id, white_config_id))
            
            cursor.execute("""
                INSERT OR REPLACE INTO game_participants (game_id, player_color, config_id)
                VALUES (?, 'black', ?)
            """, (game_id, black_config_id))
            
            conn.commit()
            conn.close()
            
            if self.logger:
                self.logger.info(f"Started game tracking: {game_id}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error starting game: {e}")
            raise
    
    def add_enhanced_move_metric(self, **kwargs) -> None:
        """
        Add a comprehensive move metric record
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build the INSERT statement dynamically based on provided fields
            fields = []
            values = []
            placeholders = []
            
            # Essential fields (always required)
            essential_fields = [
                'game_id', 'move_number', 'player_color', 'move_uci', 
                'fen_before', 'search_algorithm', 'time_taken'
            ]
            
            for field in essential_fields:
                if field in kwargs:
                    fields.append(field)
                    values.append(kwargs[field])
                    placeholders.append('?')
                else:
                    # Provide default values for missing essential fields
                    fields.append(field)
                    if field == 'time_taken':
                        values.append(0.0)
                    elif field in ['move_number', 'nodes_searched', 'depth_reached']:
                        values.append(0)
                    else:
                        values.append('')
                    placeholders.append('?')
            
            # Optional fields (add if provided)
            optional_fields = [
                'move_san', 'fen_after', 'depth_reached', 'nodes_searched', 'nps',
                'branching_factor', 'evaluation', 'mate_in', 'best_line',
                'game_phase', 'position_type', 'material_balance', 'piece_count',
                'from_opening_book', 'opening_book_name', 'move_type',
                'is_check', 'is_checkmate', 'gives_check',
                # All the detailed scoring fields
                'total_score', 'material_score', 'pst_score',
                'checkmate_threats_score', 'king_safety_score', 'king_threat_score',
                'king_endangerment_score', 'draw_scenarios_score', 'piece_coordination_score',
                'center_control_score', 'pawn_structure_score', 'pawn_weaknesses_score',
                'passed_pawns_score', 'pawn_majority_score', 'bishop_pair_score',
                'knight_pair_score', 'bishop_vision_score', 'rook_coordination_score',
                'castling_protection_score', 'castling_sacrifice_score', 'piece_activity_score',
                'improved_minor_piece_activity_score', 'mobility_score', 'undeveloped_pieces_score',
                'hanging_pieces_score', 'undefended_pieces_score', 'queen_capture_score',
                'tempo_modifier_score', 'en_passant_score', 'open_files_score', 'stalemate_score',
                # V7P3R Scoring Dictionary Fields
                'scoring_fen', 'scoring_move', 'scoring_score', 'scoring_game_phase', 'scoring_endgame_factor',
                'scoring_checkmate_threats', 'scoring_king_safety', 'scoring_king_attack', 'scoring_draw_scenarios',
                'scoring_material_score', 'scoring_piece_coordination', 'scoring_center_control', 
                'scoring_pawn_structure', 'scoring_pawn_weaknesses', 'scoring_passed_pawns', 'scoring_pawn_count',
                'scoring_pawn_promotion', 'scoring_bishop_count', 'scoring_knight_count', 'scoring_bishop_vision',
                'scoring_rook_coordination', 'scoring_castling', 'scoring_castling_protection',
                # Metadata
                'engine_config_id', 'exclude_from_analysis', 'notes'
            ]
            
            for field in optional_fields:
                if field in kwargs:
                    fields.append(field)
                    values.append(kwargs[field])
                    placeholders.append('?')
            
            # Add timestamp
            fields.append('created_at')
            values.append(datetime.now().isoformat())
            placeholders.append('?')
            
            # Execute the insert
            sql = f"""
                INSERT INTO move_metrics ({', '.join(fields)})
                VALUES ({', '.join(placeholders)})
            """
            
            print(f"DEBUG DB: Executing SQL with {len(values)} values")
            print(f"DEBUG DB: Game ID: {kwargs.get('game_id', 'unknown')}")
            
            cursor.execute(sql, values)
            conn.commit()
            conn.close()
            
            print(f"DEBUG DB: Successfully inserted move metric")
            
            if self.logger:
                self.logger.info(f"Added enhanced move metric for game {kwargs.get('game_id', 'unknown')}")
                
        except Exception as e:
            print(f"DEBUG DB ERROR: {e}")
            import traceback
            traceback.print_exc()
            if self.logger:
                self.logger.error(f"Error adding enhanced move metric: {e}")
            raise
    
    def finish_game(self, game_id: str, result: str, termination: Optional[str] = None, 
                total_moves: Optional[int] = None, game_duration: Optional[float] = None) -> None:
        """
        Update game record with final results
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            update_fields = ['result = ?']
            update_values = [result]
            
            if termination:
                update_fields.append('termination = ?')
                update_values.append(termination)
            
            if total_moves is not None:
                update_fields.append('total_moves = ?')
                update_values.append(str(total_moves))
            
            if game_duration is not None:
                update_fields.append('game_duration = ?')
                update_values.append(str(game_duration))
            
            update_values.append(game_id)
            
            cursor.execute(f"""
                UPDATE games 
                SET {', '.join(update_fields)}
                WHERE game_id = ?
            """, update_values)
            
            conn.commit()
            conn.close()
            
            if self.logger:
                self.logger.info(f"Finished game: {game_id} with result {result}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error finishing game: {e}")
            raise
    
    def get_game_summary(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive game summary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM game_summary WHERE game_id = ?
            """, (game_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            
            return None
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting game summary: {e}")
            return None
    
    def get_engine_performance(self, engine_name: Optional[str] = None, engine_version: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get engine performance statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            sql = "SELECT * FROM engine_performance"
            params = []
            
            if engine_name or engine_version:
                conditions = []
                if engine_name:
                    conditions.append("engine_name = ?")
                    params.append(engine_name)
                if engine_version:
                    conditions.append("engine_version = ?")
                    params.append(engine_version)
                sql += " WHERE " + " AND ".join(conditions)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            if rows:
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
            
            conn.close()
            return []
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting engine performance: {e}")
            return []
    
    def backup_current_database(self, backup_path: Optional[str] = None) -> Optional[str]:
        """
        Create a backup of the current database
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"metrics/chess_metrics_backup_{timestamp}.db"
        
        try:
            if os.path.exists(self.db_path):
                import shutil
                shutil.copy2(self.db_path, backup_path)
                
                if self.logger:
                    self.logger.info(f"Database backup created: {backup_path}")
                
                return backup_path
            else:
                if self.logger:
                    self.logger.warning("No database to backup")
                return None
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating backup: {e}")
            raise
    
    def migrate_legacy_data(self, legacy_db_path: str) -> None:
        """
        Migrate data from the legacy database format
        """
        try:
            if not os.path.exists(legacy_db_path):
                if self.logger:
                    self.logger.warning(f"Legacy database not found: {legacy_db_path}")
                return
            
            legacy_conn = sqlite3.connect(legacy_db_path)
            legacy_cursor = legacy_conn.cursor()
            
            # Get all games from legacy database
            legacy_cursor.execute("SELECT DISTINCT game_id FROM move_metrics ORDER BY game_id")
            game_ids = [row[0] for row in legacy_cursor.fetchall()]
            
            migrated_games = 0
            migrated_moves = 0
            
            for game_id in game_ids:
                # Get game moves
                legacy_cursor.execute("""
                    SELECT * FROM move_metrics 
                    WHERE game_id = ? 
                    ORDER BY move_number
                """, (game_id,))
                
                moves = legacy_cursor.fetchall()
                if not moves:
                    continue
                
                # Extract game information from first move
                first_move = moves[0]
                # Assume columns based on legacy schema inspection
                # This would need to be adjusted based on actual legacy schema
                
                # Create a basic game record
                try:
                    white_player = "unknown"
                    black_player = "unknown"
                    
                    # Create basic engine configs
                    basic_config = {
                        'name': 'legacy_engine',
                        'version': '1.0.0',
                        'search_algorithm': 'unknown',
                        'depth': 3
                    }
                    
                    self.start_game(game_id, white_player, black_player, basic_config, basic_config)
                    
                    # Migrate moves
                    for move_data in moves:
                        # Map legacy fields to new format
                        enhanced_move = {
                            'game_id': game_id,
                            'move_number': move_data[2] if len(move_data) > 2 else 0,
                            'player_color': 'white' if move_data[3] == 'w' else 'black' if len(move_data) > 3 else 'white',
                            'move_uci': move_data[4] if len(move_data) > 4 else '',
                            'fen_before': move_data[5] if len(move_data) > 5 else '',
                            'evaluation': move_data[6] if len(move_data) > 6 else 0.0,
                            'search_algorithm': move_data[7] if len(move_data) > 7 else 'unknown',
                            'depth_reached': move_data[8] if len(move_data) > 8 else 0,
                            'nodes_searched': move_data[9] if len(move_data) > 9 else 0,
                            'time_taken': move_data[10] if len(move_data) > 10 else 0.0,
                            'best_line': move_data[11] if len(move_data) > 11 else '',
                            'notes': 'Migrated from legacy database'
                        }
                        
                        self.add_enhanced_move_metric(**enhanced_move)
                        migrated_moves += 1
                    
                    migrated_games += 1
                    
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error migrating game {game_id}: {e}")
                    continue
            
            legacy_conn.close()
            
            if self.logger:
                self.logger.info(f"Migration completed: {migrated_games} games, {migrated_moves} moves")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during migration: {e}")
            raise

if __name__ == "__main__":
    # Test the enhanced metrics store
    store = EnhancedMetricsStore()
    print("Enhanced Metrics Store initialized successfully!")
    print(f"Database path: {store.db_path}")
    
    # Test adding a configuration
    test_config = {
        'name': 'v7p3r',
        'version': '2.0.0',
        'search_algorithm': 'minimax',
        'depth': 3,
        'max_depth': 5,
        'ruleset': 'enhanced_ruleset'
    }
    
    config_id = store.add_engine_config(test_config)
    print(f"Test configuration added with ID: {config_id}")
