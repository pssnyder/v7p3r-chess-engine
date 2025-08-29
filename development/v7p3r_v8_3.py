#!/usr/bin/env python3
"""
V7P3R Chess Engine v8.3 - "Memory Optimization & Performance Auditing"
Advanced memory management with intelligent caching and comprehensive performance monitoring
Building on V8.2's enhanced move ordering with optimized resource usage
Author: Pat Snyder
"""

import chess
import chess.engine
import time
import sys
import psutil
from typing import Optional, Tuple, Dict, Any, List, Set
from dataclasses import dataclass
from v7p3r_scoring_calculation import V7P3RScoringCalculationClean
from v7p3r_memory_manager import V7P3RMemoryManager, MemoryPolicy, create_memory_manager
from v7p3r_performance_monitor import PerformanceProfiler, profile, profiled_section


@dataclass
class SearchOptions:
    """Configuration options for the unified search"""
    return_pv: bool = True
    use_killer_moves: bool = True
    use_history_heuristic: bool = True
    use_late_move_reduction: bool = True
    use_null_move_pruning: bool = False


@dataclass
class GamePhase:
    """Game phase detection for contextual move ordering"""
    is_opening: bool = False
    is_middlegame: bool = False
    is_endgame: bool = False
    moves_played: int = 0
    pieces_developed: int = 0


@dataclass
class MoveOrderingContext:
    """Cached context for efficient move ordering"""
    has_captures: bool = False
    capture_count: int = 0
    king_in_danger: bool = False
    tactical_opportunities: Optional[List[chess.Square]] = None
    enemy_piece_positions: Optional[Dict[chess.PieceType, List[chess.Square]]] = None
    
    def __post_init__(self):
        if self.tactical_opportunities is None:
            self.tactical_opportunities = []
        if self.enemy_piece_positions is None:
            self.enemy_piece_positions = {}


class V7P3RCleanEngine:
    """V8.3 - Memory Optimization & Performance Auditing"""
    
    def __init__(self, max_memory_mb: float = 100.0):
        # Basic configuration
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300, 
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0  # King safety handled separately
        }
        
        # Search configuration
        self.default_depth = 6
        self.nodes_searched = 0
        
        # Evaluation components
        self.scoring_calculator = V7P3RScoringCalculationClean(self.piece_values)
        
        # V8.3: Advanced memory management system
        self.memory_manager = create_memory_manager(max_memory_mb, "middlegame")
        
        # V8.3: Performance monitoring and optimization
        self.profiler = PerformanceProfiler()
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Performance statistics (enhanced from V8.2)
        self.search_stats = {
            'nodes_per_second': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_cleanups': 0,
            'pressure_cleanups': 0,
            'avg_search_time': 0.0,
            'peak_memory_mb': self.baseline_memory
        }
        
        # V8.3: Game phase tracking for dynamic optimization
        self.current_game_phase = "opening"
        self.move_count = 0
        
        # Legacy compatibility - these now use memory manager
        self.killer_moves = {}  # Will be migrated to memory manager
        self.history_scores = {}  # Will be migrated to memory manager
        self.evaluation_cache = {}  # Will be migrated to memory manager
        self.transposition_table = {}  # Will be migrated to memory manager
        
        # Initialize with opening knowledge
        self._inject_opening_knowledge()
        
        # V8.3: Performance optimization features
        self._setup_performance_monitoring()
    
    def _setup_performance_monitoring(self):
        """Setup comprehensive performance monitoring"""
        # Profile critical functions
        self.search = self.profiler.profile_function(self.__class__.search)
        self._unified_search_root = self.profiler.profile_function(self.__class__._unified_search_root)
        self._minimax_alphabeta = self.profiler.profile_function(self.__class__._minimax_alphabeta)
        self._order_moves_enhanced = self.profiler.profile_function(self.__class__._order_moves_enhanced)
        self.evaluate_position = self.profiler.profile_function(self.__class__.evaluate_position)
    
    @profile
    def search(self, board: chess.Board, time_limit: float = 3.0) -> chess.Move:
        """Main search entry point with memory-optimized evaluation"""
        with profiled_section("search_initialization"):
            print("info string Starting search...", flush=True)
            sys.stdout.flush()
            
            # V8.3: Memory and performance monitoring
            search_start_memory = self.process.memory_info().rss / 1024 / 1024
            
            self.nodes_searched = 0
            start_time = time.time()
            
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return chess.Move.null()
            
            # V8.3: Dynamic game phase detection and memory optimization
            self._update_game_phase(board)
            self._optimize_memory_for_phase()
        
        with profiled_section("search_configuration"):
            # Enhanced time management
            target_time = min(time_limit * 0.6, 8.0)
            max_time = min(time_limit * 0.8, 12.0)
            
            # Configure search options based on available time and memory
            search_options = self._configure_search_options(time_limit)
        
        with profiled_section("iterative_deepening"):
            # Unified iterative deepening with memory monitoring
            best_move = legal_moves[0]
            best_pv = [best_move]
            depth = 1
            
            while depth <= self.default_depth:
                iteration_start = time.time()
                iteration_start_memory = self.process.memory_info().rss / 1024 / 1024
                
                try:
                    with profiled_section(f"depth_{depth}_search"):
                        move, score, pv = self._unified_search_root(board, depth, search_options)
                    
                    iteration_time = time.time() - iteration_start
                    iteration_memory = self.process.memory_info().rss / 1024 / 1024
                    
                    if move:
                        best_move = move
                        best_pv = pv
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        nps = int(self.nodes_searched / max(elapsed_ms / 1000, 0.001))
                        pv_str = " ".join(str(m) for m in pv[:depth])
                        
                        # UCI score from side-to-move perspective
                        score_str = self._format_uci_score(score, depth)
                        
                        print(f"info depth {depth} score {score_str} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {pv_str}")
                        sys.stdout.flush()
                    
                    # V8.3: Memory pressure detection and cleanup
                    memory_increase = iteration_memory - iteration_start_memory
                    if memory_increase > 5.0:  # More than 5MB increase in one iteration
                        cleanup_count = self.memory_manager.pressure_cleanup()
                        self.search_stats['pressure_cleanups'] += 1
                        print(f"info string Memory pressure cleanup: removed {cleanup_count} entries")
                    
                    # V8.3: Enhanced time management with memory considerations
                    elapsed = time.time() - start_time
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    
                    # Stop early if memory usage is critical
                    if current_memory > self.memory_manager.policy.critical_memory_mb:
                        print("info string Stopping early due to memory pressure")
                        break
                    
                    if iteration_time > target_time * 0.4 or elapsed > target_time:
                        break
                        
                    if elapsed + (iteration_time * 2.5) > max_time:
                        break
                        
                    depth += 1
                except Exception as e:
                    print(f"info string Search error at depth {depth}: {str(e)}")
                    break
        
        with profiled_section("search_finalization"):
            # V8.3: Update performance statistics
            total_time = time.time() - start_time
            final_memory = self.process.memory_info().rss / 1024 / 1024
            memory_used = final_memory - search_start_memory
            
            self.search_stats['avg_search_time'] = (
                self.search_stats.get('avg_search_time', 0) * 0.9 + total_time * 0.1
            )
            self.search_stats['peak_memory_mb'] = max(
                self.search_stats['peak_memory_mb'], final_memory
            )
            
            # Routine cleanup after search
            self.memory_manager.routine_cleanup()
            self.search_stats['memory_cleanups'] += 1
            
            # Log performance metrics periodically
            if self.move_count % 10 == 0:
                self._log_performance_metrics()
        
        return best_move
    
    def _update_game_phase(self, board: chess.Board):
        """V8.3: Dynamic game phase detection for memory optimization"""
        self.move_count += 1
        
        # Piece count analysis
        piece_count = len(board.piece_map())
        material_value = sum(
            self.piece_values.get(piece.piece_type, 0)
            for piece in board.piece_map().values()
        )
        
        # Development analysis
        developed_pieces = 0
        for square in [chess.B1, chess.C1, chess.F1, chess.G1,  # White
                      chess.B8, chess.C8, chess.F8, chess.G8]:  # Black
            if board.piece_at(square) is None:
                developed_pieces += 1
        
        # Phase classification
        if self.move_count <= 15 and material_value > 6000:
            self.current_game_phase = "opening"
        elif piece_count <= 12 or material_value < 2000:
            self.current_game_phase = "endgame"
        else:
            self.current_game_phase = "middlegame"
    
    def _optimize_memory_for_phase(self):
        """V8.3: Optimize memory allocation based on game phase"""
        self.memory_manager.optimize_for_game_phase(self.current_game_phase)
        
        # Phase-specific optimizations
        if self.current_game_phase == "opening":
            # Opening: Prioritize transposition table for opening knowledge
            print("info string Optimizing memory for opening phase")
        elif self.current_game_phase == "endgame":
            # Endgame: Prioritize evaluation cache for precise calculations
            print("info string Optimizing memory for endgame phase")
        elif self.current_game_phase == "middlegame":
            # Middlegame: Balanced approach with frequent cleanup
            print("info string Optimizing memory for middlegame phase")
    
    @profile
    def evaluate_position(self, board: chess.Board) -> float:
        """V8.3: Memory-optimized position evaluation with caching"""
        # Create position hash for caching
        cache_key = self._create_position_hash(board)
        
        # Check memory-managed evaluation cache
        cached_eval = self.memory_manager.get_evaluation(cache_key)
        if cached_eval is not None:
            self.search_stats['cache_hits'] += 1
            return cached_eval
        
        self.search_stats['cache_misses'] += 1
        
        # Calculate evaluation using existing logic
        final_score = self.scoring_calculator.calculate_score(board)
        
        # Store in memory-managed cache
        self.memory_manager.store_evaluation(cache_key, final_score)
        
        return final_score
    
    def _create_position_hash(self, board: chess.Board) -> str:
        """Create a hash key for position caching"""
        # Simple hash based on FEN without move counts
        fen_parts = board.fen().split(' ')
        position_fen = ' '.join(fen_parts[:4])  # Position, side, castling, ep
        return position_fen
    
    @profile
    def _unified_search_root(self, board: chess.Board, depth: int, options: SearchOptions) -> Tuple[Optional[chess.Move], float, list]:
        """V8.3: Root search with memory-optimized move ordering"""
        if depth <= 0:
            return None, self.evaluate_position(board), []
        
        with profiled_section("root_move_generation"):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                if board.is_checkmate():
                    return None, -999999, []
                else:
                    return None, 0, []
        
        with profiled_section("root_move_ordering"):
            # V8.3: Memory-efficient move ordering
            ordered_moves = self._order_moves_enhanced(board, legal_moves, 0, options)
        
        best_move = None
        best_score = float('-inf')
        best_pv = []
        alpha = float('-inf')
        beta = float('inf')
        
        with profiled_section("root_move_evaluation"):
            for move in ordered_moves:
                board.push(move)
                self.nodes_searched += 1
                
                try:
                    # V8.3: Memory-monitored recursive search
                    with profiled_section(f"move_{move}_evaluation"):
                        score, pv = self._minimax_alphabeta(board, depth - 1, alpha, beta, False, options)
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_pv = [move] + pv
                        alpha = max(alpha, score)
                    
                except Exception as e:
                    print(f"info string Error evaluating move {move}: {str(e)}")
                finally:
                    board.pop()
        
        return best_move, best_score, best_pv
    
    def _log_performance_metrics(self):
        """V8.3: Log comprehensive performance metrics"""
        memory_stats = self.memory_manager.get_memory_stats()
        current_memory = self.process.memory_info().rss / 1024 / 1024
        
        print(f"info string === V8.3 Performance Metrics ===")
        print(f"info string Memory: {current_memory:.1f}MB (peak: {self.search_stats['peak_memory_mb']:.1f}MB)")
        print(f"info string Cache hits: {self.search_stats['cache_hits']}, misses: {self.search_stats['cache_misses']}")
        print(f"info string Eval cache: {memory_stats['evaluation_cache']['size']} entries")
        print(f"info string TT entries: {memory_stats['transposition_table']['size']} entries")
        print(f"info string Memory cleanups: {self.search_stats['memory_cleanups']}")
        print(f"info string Game phase: {self.current_game_phase}")
        
        # V8.3: Performance recommendations
        if self.move_count % 20 == 0:
            recommendations = self.profiler.get_optimization_recommendations()
            if recommendations:
                print(f"info string Performance recommendations available: {len(recommendations)}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """V8.3: Get comprehensive performance analysis"""
        profiler_report = self.profiler.get_performance_report()
        memory_stats = self.memory_manager.get_memory_stats()
        
        return {
            'engine_version': 'V8.3',
            'game_phase': self.current_game_phase,
            'move_count': self.move_count,
            'search_stats': self.search_stats,
            'memory_stats': memory_stats,
            'profiler_report': profiler_report,
            'optimization_recommendations': self.profiler.get_optimization_recommendations()
        }
    
    # === Rest of the methods remain the same as V8.2 ===
    # (This is just the V8.3-specific memory and performance additions)
    
    def _configure_search_options(self, time_limit: float) -> SearchOptions:
        """Configure search options based on time constraints and memory"""
        # V8.3: Consider memory pressure in search configuration
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_pressure = current_memory > self.memory_manager.policy.memory_pressure_mb
        
        return SearchOptions(
            return_pv=True,
            use_killer_moves=not memory_pressure,  # Disable under memory pressure
            use_history_heuristic=time_limit > 1.0,
            use_late_move_reduction=time_limit > 2.0,
            use_null_move_pruning=False  # Keep disabled for consistency
        )
    
    def _inject_opening_knowledge(self):
        """Load basic opening principles into memory-managed transposition table"""
        opening_positions = {
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": {
                "best_move": "e7e5",
                "evaluation": 0,
                "depth": 4,
                "note": "King's pawn opening"
            },
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2": {
                "best_move": "g1f3",
                "evaluation": 0,
                "depth": 4,
                "note": "Develop knight"
            }
        }
        
        for fen, data in opening_positions.items():
            self.memory_manager.store_transposition(fen, data)


# Additional V8.3 methods would be imported from V8.2 base class
# This file focuses on the memory management and performance monitoring additions
