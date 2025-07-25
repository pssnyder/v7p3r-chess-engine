#!/usr/bin/env python3
"""
Enhanced Database Schema Design for V7P3R Chess Engine Metrics System
This module defines the new database structure for comprehensive chess engine analysis
"""

ENHANCED_CHESS_METRICS_SCHEMA = """
-- =============================================
-- ENHANCED CHESS METRICS DATABASE SCHEMA v2.0
-- =============================================

-- Games table - Core game information
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT UNIQUE NOT NULL,
    pgn_filename TEXT,
    timestamp TEXT NOT NULL,
    white_player TEXT NOT NULL,
    black_player TEXT NOT NULL,
    result TEXT, -- '1-0', '0-1', '1/2-1/2', '*'
    termination TEXT, -- checkmate, stalemate, resignation, etc.
    total_moves INTEGER,
    game_duration REAL, -- Total game time in seconds
    opening_name TEXT,
    eco_code TEXT, -- ECO opening classification
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Engine configurations - Track engine settings for each game
CREATE TABLE IF NOT EXISTS engine_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id TEXT UNIQUE NOT NULL, -- Hash of configuration
    engine_name TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    search_algorithm TEXT NOT NULL,
    depth INTEGER,
    max_depth INTEGER,
    max_moves INTEGER,
    ruleset TEXT,
    use_game_phase BOOLEAN,
    time_control TEXT, -- JSON string of time control settings
    other_settings TEXT, -- JSON string of additional settings
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Game participants - Links games to engine configurations
CREATE TABLE IF NOT EXISTS game_participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    player_color TEXT NOT NULL, -- 'white' or 'black'
    config_id TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (config_id) REFERENCES engine_configs(config_id)
);

-- Enhanced move metrics with detailed scoring breakdown
CREATE TABLE IF NOT EXISTS move_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    move_number INTEGER NOT NULL,
    player_color TEXT NOT NULL, -- 'white' or 'black'
    move_san TEXT NOT NULL, -- Standard algebraic notation
    move_uci TEXT NOT NULL, -- UCI notation
    fen_before TEXT NOT NULL,
    fen_after TEXT NOT NULL,
    
    -- Search metrics
    search_algorithm TEXT NOT NULL,
    depth_reached INTEGER,
    nodes_searched INTEGER DEFAULT 0,
    time_taken REAL NOT NULL,
    nps REAL, -- Nodes per second
    branching_factor REAL,
    
    -- Engine evaluation
    evaluation REAL,
    mate_in INTEGER, -- If mate detected, moves to mate
    best_line TEXT, -- Principal variation
    
    -- Game phase and position classification
    game_phase TEXT, -- 'opening', 'middlegame', 'endgame'
    position_type TEXT, -- 'tactical', 'positional', 'balanced'
    material_balance REAL,
    piece_count INTEGER,
    
    -- Opening book usage
    from_opening_book BOOLEAN DEFAULT FALSE,
    opening_book_name TEXT,
    
    -- Move classification
    move_type TEXT, -- 'normal', 'capture', 'castling', 'promotion', 'en_passant'
    is_check BOOLEAN DEFAULT FALSE,
    is_checkmate BOOLEAN DEFAULT FALSE,
    gives_check BOOLEAN DEFAULT FALSE,
    
    -- Detailed v7p3r scoring breakdown (when applicable)
    total_score REAL,
    material_score REAL,
    pst_score REAL, -- Piece-square table score
    checkmate_threats_score REAL,
    king_safety_score REAL,
    king_threat_score REAL,
    king_endangerment_score REAL,
    draw_scenarios_score REAL,
    piece_coordination_score REAL,
    center_control_score REAL,
    pawn_structure_score REAL,
    pawn_weaknesses_score REAL,
    passed_pawns_score REAL,
    pawn_majority_score REAL,
    bishop_pair_score REAL,
    knight_pair_score REAL,
    bishop_vision_score REAL,
    rook_coordination_score REAL,
    castling_protection_score REAL,
    castling_sacrifice_score REAL,
    piece_activity_score REAL,
    improved_minor_piece_activity_score REAL,
    mobility_score REAL,
    undeveloped_pieces_score REAL,
    hanging_pieces_score REAL,
    undefended_pieces_score REAL,
    queen_capture_score REAL,
    tempo_modifier_score REAL,
    en_passant_score REAL,
    open_files_score REAL,
    stalemate_score REAL,
    
    -- V7P3R Scoring Dictionary Fields (from self.scoring in v7p3r_score.py)
    scoring_fen TEXT, -- FEN from scoring dictionary
    scoring_move TEXT, -- Move from scoring dictionary (string representation)
    scoring_score REAL, -- Score from scoring dictionary
    scoring_game_phase TEXT, -- Game phase from scoring dictionary
    scoring_endgame_factor REAL, -- Endgame factor from scoring dictionary
    scoring_checkmate_threats REAL, -- Individual scoring components
    scoring_king_safety REAL,
    scoring_king_attack REAL,
    scoring_draw_scenarios REAL,
    scoring_material_score REAL,
    scoring_piece_coordination REAL,
    scoring_center_control REAL,
    scoring_pawn_structure REAL,
    scoring_pawn_weaknesses REAL,
    scoring_passed_pawns REAL,
    scoring_pawn_count REAL,
    scoring_pawn_promotion REAL,
    scoring_bishop_count REAL,
    scoring_knight_count REAL,
    scoring_bishop_vision REAL,
    scoring_rook_coordination REAL,
    scoring_castling REAL,
    scoring_castling_protection REAL,
    
    -- Technical metadata
    engine_config_id TEXT,
    exclude_from_analysis BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (engine_config_id) REFERENCES engine_configs(config_id)
);

-- Position analysis - Detailed position characteristics
CREATE TABLE IF NOT EXISTS position_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    move_number INTEGER NOT NULL,
    fen TEXT NOT NULL,
    
    -- Material counts
    white_pawns INTEGER,
    white_knights INTEGER,
    white_bishops INTEGER,
    white_rooks INTEGER,
    white_queens INTEGER,
    black_pawns INTEGER,
    black_knights INTEGER,
    black_bishops INTEGER,
    black_rooks INTEGER,
    black_queries INTEGER,
    
    -- Position characteristics
    king_safety_white REAL,
    king_safety_black REAL,
    center_control_white REAL,
    center_control_black REAL,
    space_advantage REAL, -- Positive for white advantage
    pawn_structure_score REAL,
    piece_activity_white REAL,
    piece_activity_black REAL,
    
    -- Tactical elements
    pins_count INTEGER,
    forks_count INTEGER,
    skewers_count INTEGER,
    discovered_attacks_count INTEGER,
    tactical_motifs TEXT, -- JSON array of detected tactical patterns
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- Search efficiency metrics
CREATE TABLE IF NOT EXISTS search_efficiency (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    move_number INTEGER NOT NULL,
    player_color TEXT NOT NULL,
    
    -- Search statistics
    total_nodes INTEGER,
    nodes_per_second REAL,
    effective_branching_factor REAL,
    search_depth_reached INTEGER,
    search_depth_target INTEGER,
    time_allocated REAL,
    time_used REAL,
    time_efficiency REAL, -- time_used / time_allocated
    
    -- Search quality indicators
    principal_variation_length INTEGER,
    search_stability REAL, -- How much eval changed during search
    move_ordering_efficiency REAL,
    transposition_table_hits INTEGER,
    
    -- Algorithm-specific metrics
    alpha_beta_cutoffs INTEGER,
    quiescence_nodes INTEGER,
    extensions_used INTEGER,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- Performance benchmarks
CREATE TABLE IF NOT EXISTS performance_benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    engine_config_id TEXT NOT NULL,
    
    -- Test suite results
    test_suite_name TEXT,
    positions_tested INTEGER,
    correct_moves INTEGER,
    accuracy_percentage REAL,
    average_time_per_position REAL,
    average_nodes_per_position REAL,
    
    -- Specific benchmark metrics
    tactical_score INTEGER,
    positional_score INTEGER,
    endgame_score INTEGER,
    
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (engine_config_id) REFERENCES engine_configs(config_id)
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_move_metrics_game_id ON move_metrics(game_id);
CREATE INDEX IF NOT EXISTS idx_move_metrics_player_color ON move_metrics(player_color);
CREATE INDEX IF NOT EXISTS idx_move_metrics_engine_config ON move_metrics(engine_config_id);
CREATE INDEX IF NOT EXISTS idx_games_timestamp ON games(timestamp);
CREATE INDEX IF NOT EXISTS idx_games_players ON games(white_player, black_player);
CREATE INDEX IF NOT EXISTS idx_position_analysis_fen ON position_analysis(fen);
CREATE INDEX IF NOT EXISTS idx_search_efficiency_game ON search_efficiency(game_id, move_number);

-- Views for common queries
CREATE VIEW IF NOT EXISTS game_summary AS
SELECT 
    g.game_id,
    g.white_player,
    g.black_player,
    g.result,
    g.total_moves,
    g.game_duration,
    COUNT(mm.id) as recorded_moves,
    AVG(CASE WHEN mm.player_color = 'white' THEN mm.time_taken END) as white_avg_time,
    AVG(CASE WHEN mm.player_color = 'black' THEN mm.time_taken END) as black_avg_time,
    AVG(CASE WHEN mm.player_color = 'white' THEN mm.nodes_searched END) as white_avg_nodes,
    AVG(CASE WHEN mm.player_color = 'black' THEN mm.nodes_searched END) as black_avg_nodes
FROM games g
LEFT JOIN move_metrics mm ON g.game_id = mm.game_id
GROUP BY g.game_id;

CREATE VIEW IF NOT EXISTS engine_performance AS
SELECT 
    ec.engine_name,
    ec.engine_version,
    ec.search_algorithm,
    COUNT(DISTINCT gp.game_id) as games_played,
    AVG(mm.time_taken) as avg_time_per_move,
    AVG(mm.nodes_searched) as avg_nodes_per_move,
    AVG(mm.evaluation) as avg_evaluation,
    COUNT(CASE WHEN g.result = '1-0' AND gp.player_color = 'white' THEN 1 
               WHEN g.result = '0-1' AND gp.player_color = 'black' THEN 1 END) as wins,
    COUNT(CASE WHEN g.result = '1/2-1/2' THEN 1 END) as draws,
    COUNT(CASE WHEN g.result = '1-0' AND gp.player_color = 'black' THEN 1 
               WHEN g.result = '0-1' AND gp.player_color = 'white' THEN 1 END) as losses
FROM engine_configs ec
JOIN game_participants gp ON ec.config_id = gp.config_id
JOIN games g ON gp.game_id = g.game_id
LEFT JOIN move_metrics mm ON g.game_id = mm.game_id AND 
    ((gp.player_color = 'white' AND mm.player_color = 'white') OR 
     (gp.player_color = 'black' AND mm.player_color = 'black'))
GROUP BY ec.engine_name, ec.engine_version, ec.search_algorithm;
"""

# Data structure for enhanced metrics collection
ENHANCED_MOVE_METRICS_FIELDS = {
    # Basic move information
    'game_id': str,
    'move_number': int,
    'player_color': str,  # 'white' or 'black'
    'move_san': str,
    'move_uci': str,
    'fen_before': str,
    'fen_after': str,
    
    # Search metrics
    'search_algorithm': str,
    'depth_reached': int,
    'nodes_searched': int,
    'time_taken': float,
    'nps': float,  # nodes per second
    'branching_factor': float,
    
    # Engine evaluation
    'evaluation': float,
    'mate_in': int,
    'best_line': str,
    
    # Game phase and position
    'game_phase': str,
    'position_type': str,
    'material_balance': float,
    'piece_count': int,
    
    # Opening book
    'from_opening_book': bool,
    'opening_book_name': str,
    
    # Move classification
    'move_type': str,
    'is_check': bool,
    'is_checkmate': bool,
    'gives_check': bool,
    
    # Detailed v7p3r scoring (when applicable)
    'total_score': float,
    'material_score': float,
    'pst_score': float,
    'checkmate_threats_score': float,
    'king_safety_score': float,
    'king_threat_score': float,
    'king_endangerment_score': float,
    'draw_scenarios_score': float,
    'piece_coordination_score': float,
    'center_control_score': float,
    'pawn_structure_score': float,
    'pawn_weaknesses_score': float,
    'passed_pawns_score': float,
    'pawn_majority_score': float,
    'bishop_pair_score': float,
    'knight_pair_score': float,
    'bishop_vision_score': float,
    'rook_coordination_score': float,
    'castling_protection_score': float,
    'castling_sacrifice_score': float,
    'piece_activity_score': float,
    'improved_minor_piece_activity_score': float,
    'mobility_score': float,
    'undeveloped_pieces_score': float,
    'hanging_pieces_score': float,
    'undefended_pieces_score': float,
    'queen_capture_score': float,
    'tempo_modifier_score': float,
    'en_passant_score': float,
    'open_files_score': float,
    'stalemate_score': float,
    
    # Metadata
    'engine_config_id': str,
    'exclude_from_analysis': bool,
    'notes': str
}

if __name__ == "__main__":
    print("Enhanced Chess Metrics Database Schema v2.0")
    print("=" * 50)
    print("This schema provides comprehensive data collection for:")
    print("1. Detailed scoring breakdown (23+ evaluation components)")
    print("2. Engine-specific metrics with proper attribution")
    print("3. Game phase and position analysis")
    print("4. Search efficiency and performance metrics")
    print("5. Enhanced analytics and visualization capabilities")
    print("\nReady for implementation!")
