# v7p3r_config_types.py

"""Type definitions for V7P3R chess engine configuration."""

from typing import TypedDict, List, Optional, Union, Literal, Dict, Any

class GameConfig(TypedDict, total=False):
    """Game configuration type with optional fields."""
    game_count: int
    starting_position: str
    white_player: str
    black_player: str
    time_control: bool
    game_time: int
    time_increment: int

class EngineConfig(TypedDict, total=False):
    """Engine configuration type with optional fields."""
    # Core settings
    engine_id: str
    name: str
    version: str
    ruleset: str
    search_algorithm: Literal["minimax", "alphabeta", "negamax"]
    depth: int
    max_depth: int
    multi_threaded: bool

    # Search optimizations
    use_transpositions: bool
    use_tablebases: bool
    use_ponder: bool
    use_iterative_deepening: bool
    use_transposition_table: bool
    use_checkmate_detection: bool
    use_stalemate_detection: bool
    use_draw_prevention: bool
    use_game_phase: bool
    use_quiescence: bool
    use_mvv_lva: bool
    use_primary_scoring: bool
    use_secondary_scoring: bool
    use_ab_pruning: bool
    use_move_ordering: bool
    use_null_move: bool
    null_move_threshold: int
    use_late_move_reduction: bool
    late_move_reduction_factor: int
    max_ordered_moves: int
    mvv_lva_settings: Dict[str, Union[bool, int]]
    use_dtm: bool
    dtm_depth: int
    dtm_window: int
    use_aspiration: bool
    aspiration_window: int
    use_futility_pruning: bool
    futility_margin: int

    # Positional evaluation
    use_extended_pawn_structure: bool
    pawn_structure_weight: int
    use_king_safety: bool
    king_safety_weight: int
    use_piece_activity: bool
    piece_activity_weight: int
    use_space_control: bool
    space_control_weight: int
    use_mobility: bool
    mobility_weight: int
    use_material_balance: bool
    material_balance_weight: int
    use_attack_defense: bool
    attack_defense_weight: int
    use_blockade: bool
    blockade_weight: int
    use_open_files: bool
    open_files_weight: int
    use_semi_open_files: bool
    semi_open_files_weight: int
    use_passed_pawns: bool
    passed_pawns_weight: int
    use_isolated_pawns: bool
    isolated_pawns_weight: int
    use_backwards_pawns: bool
    backwards_pawns_weight: int
    use_doubled_pawns: bool
    doubled_pawns_weight: int
    use_weak_squares: bool
    weak_squares_weight: int
    use_strong_squares: bool
    strong_squares_weight: int
    use_color_complexity: bool
    color_complexity_weight: int
    use_piece_swarm: bool
    piece_swarm_weight: int
    use_knight_outposts: bool
    knight_outposts_weight: int
    use_bishop_pair: bool
    bishop_pair_weight: int
    use_rook_lift: bool
    rook_lift_weight: int
    use_queen_activity: bool
    queen_activity_weight: int
    use_king_activity: bool
    king_activity_weight: int
    use_castle_safety: bool
    castle_safety_weight: int
    use_risk_management: bool
    risk_management_weight: int
    use_material_count: bool
    material_count_weight: int
    use_piece_value: bool
    piece_value_weight: int

    # Distance-based evaluation
    use_king_distance: bool
    king_distance_weight: int
    use_queen_distance: bool
    queen_distance_weight: int
    use_rook_distance: bool
    rook_distance_weight: int
    use_bishop_distance: bool
    bishop_distance_weight: int
    use_knight_distance: bool
    knight_distance_weight: int
    use_pawn_distance: bool
    pawn_distance_weight: int
    use_king_pawn_distance: bool
    king_pawn_distance_weight: int
    use_queen_pawn_distance: bool
    queen_pawn_distance_weight: int
    use_rook_pawn_distance: bool
    rook_pawn_distance_weight: int
    use_bishop_pawn_distance: bool
    bishop_pawn_distance_weight: int
    use_knight_pawn_distance: bool
    knight_pawn_distance_weight: int
    use_passed_pawn_distance: bool
    passed_pawn_distance_weight: int
    use_isolated_pawn_distance: bool
    isolated_pawn_distance_weight: int
    use_backwards_pawn_distance: bool
    backwards_pawn_distance_weight: int
    use_doubled_pawn_distance: bool
    doubled_pawn_distance_weight: int
    use_weak_square_distance: bool
    weak_square_distance_weight: int
    use_strong_square_distance: bool
    strong_square_distance_weight: int
    use_color_complexity_distance: bool
    color_complexity_distance_weight: int
    use_piece_swarm_distance: bool
    piece_swarm_distance_weight: int
    use_knight_outpost_distance: bool
    knight_outpost_distance_weight: int
    use_bishop_pair_distance: bool
    bishop_pair_distance_weight: int
    use_rook_lift_distance: bool
    rook_lift_distance_weight: int
    use_queen_activity_distance: bool
    queen_activity_distance_weight: int
    use_king_activity_distance: bool
    king_activity_distance_weight: int
    use_castle_safety_distance: bool
    castle_safety_distance_weight: int
    use_risk_management_distance: bool
    risk_management_distance_weight: int
    use_material_count_distance: bool
    material_count_distance_weight: int
    use_piece_value_distance: bool
    piece_value_distance_weight: int

    # Meta-weights
    use_king_distance_weight: bool
    king_distance_weight_weight: int
    use_queen_distance_weight: bool
    queen_distance_weight_weight: int
    use_rook_distance_weight: bool
    rook_distance_weight_weight: int
    use_bishop_distance_weight: bool
    bishop_distance_weight_weight: int
    use_knight_distance_weight: bool
    knight_distance_weight_weight: int
    use_pawn_distance_weight: bool
    pawn_distance_weight_weight: int
    use_king_pawn_distance_weight: bool
    king_pawn_distance_weight_weight: int
    use_queen_pawn_distance_weight: bool
    queen_pawn_distance_weight_weight: int
    use_rook_pawn_distance_weight: bool
    rook_pawn_distance_weight_weight: int
    use_bishop_pawn_distance_weight: bool
    bishop_pawn_distance_weight_weight: int
    use_knight_pawn_distance_weight: bool
    knight_pawn_distance_weight_weight: int
    use_passed_pawn_distance_weight: bool
    passed_pawn_distance_weight_weight: int
    use_isolated_pawn_distance_weight: bool
    isolated_pawn_distance_weight_weight: int
    use_backwards_pawn_distance_weight: bool
    backwards_pawn_distance_weight_weight: int
    use_doubled_pawn_distance_weight: bool
    doubled_pawn_distance_weight_weight: int
    use_weak_square_distance_weight: bool
    weak_square_distance_weight_weight: int
    use_strong_square_distance_weight: bool
    strong_square_distance_weight_weight: int
    use_color_complexity_distance_weight: bool
    color_complexity_distance_weight_weight: int
    use_piece_swarm_distance_weight: bool
    piece_swarm_distance_weight_weight: int
    use_knight_outpost_distance_weight: bool
    knight_outpost_distance_weight_weight: int
    use_bishop_pair_distance_weight: bool
    bishop_pair_distance_weight_weight: int
    use_rook_lift_distance_weight: bool
    rook_lift_distance_weight_weight: int
    use_queen_activity_distance_weight: bool
    queen_activity_distance_weight_weight: int
    use_king_activity_distance_weight: bool
    king_activity_distance_weight_weight: int
    use_castle_safety_distance_weight: bool
    castle_safety_distance_weight_weight: int
    use_risk_management_distance_weight: bool
    risk_management_distance_weight_weight: int
    use_material_count_distance_weight: bool
    material_count_distance_weight_weight: int
    use_piece_value_distance_weight: bool
    piece_value_distance_weight_weight: int

class StockfishConfig(TypedDict):
    engine_id: str
    name: str
    stockfish_path: str
    elo_rating: int
    skill_level: int
    debug_mode: bool
    depth: int
    max_depth: int
    movetime: int
    nodes: Optional[int]
    uci_limit_strength: bool
    contempt: int
    threads: int
    hash: int
    ponder: bool
    multi_pv: int
    syzygy_path: Optional[str]
    syzygy_probe_depth: int
    uci_chess960: bool

class PuzzleDBConfig(TypedDict):
    db_path: str
    selection: dict
    adaptive_elo: dict
    maintenance: dict

class PuzzleSolverConfig(TypedDict):
    engine: dict
    tracking: dict
    integration: dict
    display: dict

class PuzzleConfig(TypedDict, total=False):
    """Puzzle configuration type with optional fields."""
    puzzle_database: PuzzleDBConfig
    puzzle_solver: PuzzleSolverConfig
    position_config: Optional[Dict[str, Any]]

class MetricsConfig(TypedDict):
    metrics_to_track: List[str]
    include_engines: List[str]
    exclude_engine_ids: List[str]
    group_by: str
    respect_exclusion_flags: bool
    default_grouping: str
    show_engine_version: bool
    show_engine_config_hash: bool

class NeuralNetworkConfig(TypedDict):
    training: dict
    move_library: dict

class GeneticAlgorithmConfig(TypedDict):
    population_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float
    elitism_rate: float
    adaptive_mutation: bool
    positions_source: str
    positions_count: int
    max_stagnation: int
    use_cuda: bool
    cuda_batch_size: int
    use_multiprocessing: bool
    max_workers: int
    use_neural_evaluator: bool
    neural_model_path: Optional[str]
    enable_cache: bool
    max_cache_size: int

class ReinforcementLearningConfig(TypedDict):
    hidden_dim: int
    dropout: float
    learning_rate: float
    clip_ratio: float
    vf_coef: float
    ent_coef: float
    max_moves: int
    batch_size: int
    episodes_per_validation: int
    reward_ruleset: str
    model_path: str
    save_frequency: int
    use_cuda: bool
    device: str
    verbose_training: bool

class V7P3RConfig(TypedDict):
    """Complete V7P3R chess engine configuration."""
    game_config: GameConfig
    engine_config: EngineConfig
    stockfish_config: StockfishConfig
    puzzle_config: PuzzleConfig
    metrics_config: MetricsConfig
    v7p3r_nn_config: NeuralNetworkConfig
    v7p3r_ga_config: GeneticAlgorithmConfig
    v7p3r_rl_config: ReinforcementLearningConfig
