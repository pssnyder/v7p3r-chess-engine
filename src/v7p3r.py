#!/usr/bin/env python3
"""
V7P3R Chess Engine v9.0 - Tournament Ready
Consolidated V8.x series improvements with memory optimization and enhanced performance
Author: Pat Snyder
"""

import time
import chess
import chess.pgn
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import OrderedDict

# V9.0 Memory Management Integration
class MemoryPolicy:
    """Memory management policy configuration"""
    max_cache_size: int = 50000  # Maximum entries in evaluation cache
    max_tt_size: int = 100000    # Maximum transposition table entries
    max_killer_moves: int = 1000  # Maximum killer move entries
    max_history_size: int = 10000 # Maximum history heuristic entries
    
    # Age-based cleanup thresholds (in seconds)
    cache_ttl: float = 30.0      # Time-to-live for cache entries
    tt_ttl: float = 60.0         # Time-to-live for transposition entries
    killer_ttl: float = 10.0     # Time-to-live for killer moves
    history_ttl: float = 20.0    # Time-to-live for history scores
    
    # Memory pressure thresholds
    memory_pressure_mb: float = 100.0  # Start cleanup at this memory usage
    critical_memory_mb: float = 200.0  # Emergency cleanup threshold
    
    # Cleanup frequencies
    cleanup_interval: float = 5.0      # Seconds between routine cleanups
    pressure_cleanup_ratio: float = 0.3 # Fraction to remove under pressure


class LRUCacheWithTTL:
    """LRU cache with time-to-live support for memory-efficient storage"""
    
    def __init__(self, max_size: int, ttl: float):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.access_count = 0
        self.hit_count = 0
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value with LRU update and TTL check"""
        self.access_count += 1
        current_time = time.time()
        
        if key in self.cache:
            # Check if entry has expired
            if current_time - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hit_count += 1
            return self.cache[key]
        
        return None
    
    def put(self, key: Any, value: Any):
        """Store value with automatic size management"""
        current_time = time.time()
        
        if key in self.cache:
            # Update existing entry
            self.cache[key] = value
            self.timestamps[key] = current_time
            self.cache.move_to_end(key)
        else:
            # Add new entry
            self.cache[key] = value
            self.timestamps[key] = current_time
            
            # Remove oldest entries if over size limit
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]
        
        return len(expired_keys)
    
    def cleanup_pressure(self, ratio: float) -> int:
        """Remove oldest entries under memory pressure"""
        target_removal = int(len(self.cache) * ratio)
        removed = 0
        
        keys_to_remove = list(self.cache.keys())[:target_removal]
        for key in keys_to_remove:
            del self.cache[key]
            del self.timestamps[key]
            removed += 1
        
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        hit_ratio = self.hit_count / max(self.access_count, 1)
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_ratio': hit_ratio,
            'access_count': self.access_count,
            'hit_count': self.hit_count
        }
    
    def clear(self):
        """Clear all cache data"""
        self.cache.clear()
        self.timestamps.clear()
        self.access_count = 0
        self.hit_count = 0


import chess
import chess.engine
import time
import sys
from typing import Optional, Tuple, Dict, Any, List, Set
from dataclasses import dataclass
from v7p3r_scoring_calculation import V7P3RScoringCalculationClean


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
    """V9.2 - Clean Engine with Deterministic Evaluation"""
    
    def __init__(self):
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
        
        # Unified search optimizations (kept from V8.0)
        self.killer_moves = {}  # killer_moves[ply] = [move1, move2]
        self.history_scores = {}  # history_scores[move_key] = score
        
        # Simple evaluation cache (deterministic, no async issues)
        self.evaluation_cache = {}  # position_hash -> evaluation
        
        # Performance monitoring (simplified)
        self.search_stats = {
            'nodes_per_second': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
        
        # Transposition table for opening guidance only
        self.transposition_table: Dict[str, Dict[str, Any]] = {}
        self._inject_opening_knowledge()
    
    def search(self, board: chess.Board, time_limit: float = 3.0) -> chess.Move:
        """Main search entry point with consistent evaluation"""
        print("info string Starting search...", flush=True)
        sys.stdout.flush()
        
        self.nodes_searched = 0
        start_time = time.time()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
            
        # Enhanced time management (kept from V8.0)
        target_time = min(time_limit * 0.6, 8.0)
        max_time = min(time_limit * 0.8, 12.0)
        
        # Configure search options based on available time
        search_options = self._configure_search_options(time_limit)
        
        # Unified iterative deepening (V8.0 architecture, V8.1 consistency)
        best_move = legal_moves[0]
        best_pv = [best_move]
        depth = 1
        
        while depth <= self.default_depth:
            iteration_start = time.time()
            try:
                move, score, pv = self._unified_search_root(board, depth, search_options)
                iteration_time = time.time() - iteration_start
                
                if move:
                    best_move = move
                    best_pv = pv
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    nps = int(self.nodes_searched / max(elapsed_ms / 1000, 0.001))
                    pv_str = " ".join(str(m) for m in pv[:depth])
                    
                    # UCI score from side-to-move perspective (fixed in V8.0)
                    score_str = self._format_uci_score(score, depth)
                    
                    print(f"info depth {depth} score {score_str} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {pv_str}")
                    sys.stdout.flush()
                
                # Time management (kept from V8.0)
                elapsed = time.time() - start_time
                
                if iteration_time > target_time * 0.4 or elapsed > target_time:
                    break
                    
                if elapsed + (iteration_time * 2.5) > max_time:
                    break
                    
                depth += 1
            except:
                break
                
        return best_move
    
    def _unified_search_root(self, board: chess.Board, depth: int, options: SearchOptions) -> Tuple[Optional[chess.Move], float, list]:
        """
        V9.2: Deterministic root search
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0.0, []
            
        # Enhanced move ordering (V8.2 - aligned with evaluation heuristics)
        legal_moves = self._order_moves_enhanced(board, legal_moves, 0, options)
        
        best_move = legal_moves[0]
        best_score = -99999.0
        best_pv = [best_move]
        alpha = -99999.0
        beta = 99999.0
        
        # V9.2: Use deterministic evaluation for move selection
        for move in legal_moves:
            board.push(move)
            try:
                score, pv = self._unified_negamax(board, depth - 1, -beta, -alpha, 1, options)
                score = -score
            finally:
                board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
                best_pv = [move] + pv if options.return_pv else [move]
                
            alpha = max(alpha, best_score)
        
        return best_move, best_score, best_pv
    
    def _unified_negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, 
                        ply: int, options: SearchOptions) -> Tuple[float, list]:
        """
        Unified negamax function with deterministic evaluation.
        V8.1: Removed async evaluation, restored consistent scoring.
        """
        self.nodes_searched += 1
        
        # Terminal depth - use deterministic evaluation
        if depth == 0:
            evaluation = self._evaluate_position_deterministic(board)
            pv = [] if options.return_pv else []
            return evaluation, pv
            
        # Terminal positions
        if board.is_game_over():
            if board.is_checkmate():
                return -29000.0 + ply, []
            return 0.0, []  # Stalemate
        
        # Null move pruning (if enabled and conditions are met)
        if (options.use_null_move_pruning and depth > 2 and 
            not board.is_check() and self._has_non_pawn_material(board)):
            # TODO: Implement null move pruning
            pass
        
        # Get and order legal moves (V8.2 enhanced ordering)
        legal_moves = list(board.legal_moves)
        legal_moves = self._order_moves_enhanced(board, legal_moves, ply, options)
        
        best_score = -99999.0
        best_pv = []
        
        for i, move in enumerate(legal_moves):
            # Late move reduction (if enabled)
            reduced_depth = depth - 1
            if (options.use_late_move_reduction and depth > 2 and i > 3 and 
                not board.is_capture(move) and not board.is_check()):
                reduced_depth = max(0, depth - 2)
            
            board.push(move)
            score, pv = self._unified_negamax(board, reduced_depth, -beta, -alpha, ply + 1, options)
            score = -score
            board.pop()
            
            if score > best_score:
                best_score = score
                if options.return_pv:
                    best_pv = [move] + pv
            
            alpha = max(alpha, score)
            
            if alpha >= beta:
                # Alpha-beta cutoff
                if options.use_killer_moves:
                    self._store_killer_move(move, ply)
                if options.use_history_heuristic:
                    self._update_history_score(move, depth)
                break
        
        return best_score, best_pv
    
    def _evaluate_position_deterministic(self, board: chess.Board) -> float:
        """
        V8.1: Deterministic evaluation - always returns the same score for the same position.
        Restored original V7.2 logic with V8.0 caching benefits.
        """
        position_hash = str(board.board_fen())
        cache_key = f"{position_hash}_{board.turn}"
        
        # Check evaluation cache first
        if cache_key in self.evaluation_cache:
            self.search_stats['cache_hits'] += 1
            return self.evaluation_cache[cache_key]
        
        self.search_stats['cache_misses'] += 1
        
        # Use the original, proven evaluation logic (from V7.2)
        white_score = self.scoring_calculator.calculate_score_optimized(board, chess.WHITE)
        black_score = self.scoring_calculator.calculate_score_optimized(board, chess.BLACK)
        
        # Return from current side-to-move perspective (negamax requirement)
        if board.turn == chess.WHITE:
            final_score = white_score - black_score
        else:
            final_score = black_score - white_score
        
        # Cache the result - deterministic, no async corruption
        self.evaluation_cache[cache_key] = final_score
        
        return final_score
    
    def _configure_search_options(self, time_limit: float) -> SearchOptions:
        """Configure search options based on available time (simplified from V8.0)"""
        if time_limit > 5.0:
            # Plenty of time - use all features
            return SearchOptions(
                return_pv=True,
                use_killer_moves=True,
                use_history_heuristic=True,
                use_late_move_reduction=True
            )
        elif time_limit > 1.0:
            # Moderate time - reduce some overhead
            return SearchOptions(
                return_pv=True,
                use_killer_moves=True,
                use_history_heuristic=False,
                use_late_move_reduction=True
            )
        else:
            # Time pressure - minimize overhead
            return SearchOptions(
                return_pv=False,
                use_killer_moves=False,
                use_history_heuristic=False,
                use_late_move_reduction=False
            )
    
    def _has_non_pawn_material(self, board: chess.Board) -> bool:
        """Check if the current player has non-pawn material"""
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            if board.pieces(piece_type, board.turn):
                return True
        return False
    
    # V8.2: Enhanced move ordering aligned with evaluation heuristics
    def _order_moves_enhanced(self, board: chess.Board, moves: List[chess.Move], 
                            ply: int, options: SearchOptions) -> List[chess.Move]:
        """V8.2: Contextual move ordering with dynamic heuristics and pre-pruning"""
        if len(moves) <= 1:
            return moves
        
        # Detect game phase for contextual ordering
        game_phase = self._detect_game_phase(board)
        
        # Build efficient context once
        context = self._build_move_ordering_context(board, moves)
        
        # Score moves with contextual heuristics
        scored_moves = []
        for move in moves:
            score = self._score_move_contextual(board, move, ply, options, game_phase, context)
            scored_moves.append((move, score))
        
        # Sort by score
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Pre-pruning: eliminate implausible moves
        pruned_moves = self._prune_implausible_moves(scored_moves, game_phase)
        
        return [move for move, _ in pruned_moves]
    
    def _detect_game_phase(self, board: chess.Board) -> GamePhase:
        """Fast game phase detection for contextual move ordering"""
        moves_played = len(board.move_stack)
        
        # Count developed pieces (knights and bishops not on starting squares)
        pieces_developed = 0
        for color in [chess.WHITE, chess.BLACK]:
            # Knights
            knights = board.pieces(chess.KNIGHT, color)
            starting_squares = {chess.B1, chess.G1} if color == chess.WHITE else {chess.B8, chess.G8}
            for square in knights:
                if square not in starting_squares:
                    pieces_developed += 1
            
            # Bishops
            bishops = board.pieces(chess.BISHOP, color)
            starting_squares = {chess.C1, chess.F1} if color == chess.WHITE else {chess.C8, chess.F8}
            for square in bishops:
                if square not in starting_squares:
                    pieces_developed += 1
        
        # Count total material to detect endgame
        total_material = 0
        for piece_type, value in self.piece_values.items():
            if piece_type != chess.KING:
                total_material += len(board.pieces(piece_type, chess.WHITE)) * value
                total_material += len(board.pieces(piece_type, chess.BLACK)) * value
        
        # Phase determination
        is_opening = moves_played < 20 and pieces_developed < 6
        is_endgame = total_material < 2500  # Less than ~2 rooks + 2 knights per side
        is_middlegame = not is_opening and not is_endgame
        
        return GamePhase(
            is_opening=is_opening,
            is_middlegame=is_middlegame,
            is_endgame=is_endgame,
            moves_played=moves_played,
            pieces_developed=pieces_developed
        )
    
    def _build_move_ordering_context(self, board: chess.Board, moves: List[chess.Move]) -> MoveOrderingContext:
        """Build context once to avoid redundant calculations"""
        # Count captures
        captures = [move for move in moves if board.is_capture(move)]
        has_captures = len(captures) > 0
        
        # Build enemy piece positions for tactical analysis
        enemy_color = not board.turn
        enemy_positions = {}
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            positions = list(board.pieces(piece_type, enemy_color))
            if positions:
                enemy_positions[piece_type] = positions
        
        # Detect tactical opportunities (alignment patterns)
        tactical_squares = self._detect_tactical_opportunities(board, enemy_positions)
        
        # Check if our king is in danger
        our_king = board.king(board.turn)
        king_in_danger = board.is_check() or (our_king is not None and self._king_under_pressure(board, our_king))
        
        return MoveOrderingContext(
            has_captures=has_captures,
            capture_count=len(captures),
            king_in_danger=king_in_danger,
            tactical_opportunities=tactical_squares,
            enemy_piece_positions=enemy_positions
        )
    
    def _detect_tactical_opportunities(self, board: chess.Board, enemy_positions: Dict) -> List[chess.Square]:
        """Fast detection of tactical patterns - pins, forks, skewers, multi-attacks"""
        opportunities = []
        our_color = board.turn
        
        # Get our attacking pieces
        our_pieces = {}
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            positions = list(board.pieces(piece_type, our_color))
            if positions:
                our_pieces[piece_type] = positions
        
        # Look for alignment patterns (same rank/file/diagonal)
        for piece_type, positions in our_pieces.items():
            for piece_square in positions:
                # Check for multiple enemy pieces in attack range
                attacked_enemies = []
                
                if piece_type in [chess.ROOK, chess.QUEEN]:
                    # Check ranks and files
                    for enemy_type, enemy_squares in enemy_positions.items():
                        for enemy_square in enemy_squares:
                            if (chess.square_rank(piece_square) == chess.square_rank(enemy_square) or
                                chess.square_file(piece_square) == chess.square_file(enemy_square)):
                                if board.attacks(piece_square) & chess.SquareSet([enemy_square]):
                                    attacked_enemies.append(enemy_square)
                
                if piece_type in [chess.BISHOP, chess.QUEEN]:
                    # Check diagonals
                    for enemy_type, enemy_squares in enemy_positions.items():
                        for enemy_square in enemy_squares:
                            rank_diff = abs(chess.square_rank(piece_square) - chess.square_rank(enemy_square))
                            file_diff = abs(chess.square_file(piece_square) - chess.square_file(enemy_square))
                            if rank_diff == file_diff and rank_diff > 0:
                                if board.attacks(piece_square) & chess.SquareSet([enemy_square]):
                                    attacked_enemies.append(enemy_square)
                
                if piece_type == chess.KNIGHT:
                    # Knight forks - check if attacking multiple pieces
                    attack_set = board.attacks(piece_square)
                    for enemy_type, enemy_squares in enemy_positions.items():
                        for enemy_square in enemy_squares:
                            if enemy_square in attack_set:
                                attacked_enemies.append(enemy_square)
                
                # If attacking 2+ pieces, it's a tactical opportunity
                if len(attacked_enemies) >= 2:
                    opportunities.extend(attacked_enemies)
        
        return list(set(opportunities))  # Remove duplicates
    
    def _king_under_pressure(self, board: chess.Board, king_square: chess.Square) -> bool:
        """Check if king is under tactical pressure (not just check)"""
        enemy_color = not board.turn
        
        # Count enemy pieces attacking near king
        king_area = [
            king_square + offset for offset in [-9, -8, -7, -1, 1, 7, 8, 9]
            if 0 <= king_square + offset < 64 and 
            abs(chess.square_file(king_square) - chess.square_file(king_square + offset)) <= 1
        ]
        
        attacks_on_king_area = 0
        for square in king_area:
            if square >= 0 and square < 64:
                if board.is_attacked_by(enemy_color, square):
                    attacks_on_king_area += 1
        
        return attacks_on_king_area >= 2  # King under pressure if 2+ squares attacked
    
    def _score_move_contextual(self, board: chess.Board, move: chess.Move, ply: int, 
                              options: SearchOptions, game_phase: GamePhase, 
                              context: MoveOrderingContext) -> float:
        """Contextual move scoring with dynamic heuristics based on game phase"""
        score = 0.0
        
        # 1. Mate detection (always highest priority)
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return 200000.0
        board.pop()
        
        # 2. Captures - only process if we have captures (efficiency)
        if context.has_captures and board.is_capture(move):
            score += self._score_capture_mvv_lva(board, move)
        
        # 3. Promotions (always high priority)
        if move.promotion:
            score += 90000 + self.piece_values.get(move.promotion, 0)
        
        # 4. Multi-piece attack opportunities (your tactical heuristic)
        if context.tactical_opportunities and move.to_square in context.tactical_opportunities:
            score += 15000  # High tactical priority
        
        # 5. Contextual heuristics based on game phase
        if game_phase.is_opening:
            score += self._score_opening_move(board, move)
        elif game_phase.is_middlegame:
            score += self._score_middlegame_move(board, move, context)
        elif game_phase.is_endgame:
            score += self._score_endgame_move(board, move)
        
        # 6. King safety - only if king is in danger or endgame
        if context.king_in_danger or game_phase.is_endgame:
            score += self._score_king_safety_move(board, move)
        
        # 7. Standard heuristics (with conditional processing)
        if options.use_killer_moves and ply in self.killer_moves:
            if move in self.killer_moves[ply]:
                score += 8000 - self.killer_moves[ply].index(move) * 1000
        
        if options.use_history_heuristic:
            move_key = f"{move.from_square}_{move.to_square}"
            if move_key in self.history_scores:
                score += min(self.history_scores[move_key], 4000)
        
        # 8. Checks (context-dependent priority)
        board.push(move)
        if board.is_check():
            if game_phase.is_middlegame:
                score += 6000  # Higher in middlegame
            else:
                score += 3000  # Lower in opening/endgame
        board.pop()
        
        return score
    
    def _score_capture_mvv_lva(self, board: chess.Board, move: chess.Move) -> float:
        """Efficient MVV-LVA scoring"""
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            victim_value = self.piece_values.get(victim.piece_type, 0)
            attacker_value = self.piece_values.get(attacker.piece_type, 0)
            return 100000 + (victim_value * 10) - attacker_value
        return 100000
    
    def _score_opening_move(self, board: chess.Board, move: chess.Move) -> float:
        """Opening-specific move scoring"""
        score = 0.0
        piece = board.piece_at(move.from_square)
        
        if piece:
            # Prioritize minor piece development
            if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                # Moving off back rank
                from_rank = chess.square_rank(move.from_square)
                to_rank = chess.square_rank(move.to_square)
                if ((piece.color == chess.WHITE and from_rank == 0 and to_rank > 0) or
                    (piece.color == chess.BLACK and from_rank == 7 and to_rank < 7)):
                    score += 2000  # Development bonus
                
                # Central squares bonus
                if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5, 
                                    chess.C4, chess.C5, chess.F4, chess.F5]:
                    score += 500
            
            # Castling bonus
            if piece.piece_type == chess.KING and abs(move.from_square - move.to_square) == 2:
                score += 3000  # Castling priority in opening
        
        return score
    
    def _score_middlegame_move(self, board: chess.Board, move: chess.Move, context: MoveOrderingContext) -> float:
        """Middlegame-specific move scoring"""
        score = 0.0
        
        # Tactical moves get higher priority
        piece = board.piece_at(move.from_square)
        if piece:
            # Pieces attacking multiple targets
            if context.tactical_opportunities and move.to_square in context.tactical_opportunities:
                score += 3000
            
            # Central control
            if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
                score += 300
            
            # Piece activity (moving to more active squares)
            from_attacks = len(board.attacks(move.from_square))
            board.push(move)
            to_attacks = len(board.attacks(move.to_square)) if board.piece_at(move.to_square) else 0
            board.pop()
            
            if to_attacks > from_attacks:
                score += (to_attacks - from_attacks) * 50
        
        return score
    
    def _score_endgame_move(self, board: chess.Board, move: chess.Move) -> float:
        """Endgame-specific move scoring"""
        score = 0.0
        piece = board.piece_at(move.from_square)
        
        if piece:
            if piece.piece_type == chess.KING:
                # King activity bonus
                enemy_king = board.king(not board.turn)
                if enemy_king:
                    # Moving closer to enemy king
                    from_distance = chess.square_distance(move.from_square, enemy_king)
                    to_distance = chess.square_distance(move.to_square, enemy_king)
                    if to_distance < from_distance:
                        score += 1000
                
                # Centralization
                center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
                if move.to_square in center_squares:
                    score += 800
            
            elif piece.piece_type == chess.PAWN:
                # Pawn advancement
                rank_bonus = chess.square_rank(move.to_square) if piece.color == chess.WHITE else 7 - chess.square_rank(move.to_square)
                score += rank_bonus * 100
        
        return score
    
    def _score_king_safety_move(self, board: chess.Board, move: chess.Move) -> float:
        """King safety move scoring"""
        score = 0.0
        piece = board.piece_at(move.from_square)
        our_king = board.king(board.turn)
        
        if piece and our_king:
            # Moving pieces closer to defend king
            if piece.piece_type != chess.KING:
                from_distance = chess.square_distance(move.from_square, our_king)
                to_distance = chess.square_distance(move.to_square, our_king)
                if to_distance < from_distance:
                    score += 400  # Defensive positioning
            
            # King escape moves if in danger
            elif piece.piece_type == chess.KING:
                enemy_color = not board.turn
                # Check if destination is safer
                board.push(move)
                attacks_on_new_square = len([sq for sq in chess.SQUARES 
                                           if board.is_attacked_by(enemy_color, move.to_square)])
                board.pop()
                
                attacks_on_old_square = len([sq for sq in chess.SQUARES 
                                           if board.is_attacked_by(enemy_color, move.from_square)])
                
                if attacks_on_new_square < attacks_on_old_square:
                    score += 2000  # Moving to safer square
        
        return score
    
    def _prune_implausible_moves(self, scored_moves: List[Tuple[chess.Move, float]], 
                                game_phase: GamePhase) -> List[Tuple[chess.Move, float]]:
        """Pre-pruning to eliminate implausible moves and speed up search"""
        if len(scored_moves) <= 8:
            return scored_moves  # Don't prune if we have few moves
        
        # Dynamic pruning threshold based on game phase
        if game_phase.is_opening:
            # Keep more moves in opening for development options
            keep_count = min(len(scored_moves), 15)
        elif game_phase.is_middlegame:
            # More aggressive pruning in tactical middlegame
            keep_count = min(len(scored_moves), 12)
        else:  # endgame
            # Keep precision in endgame
            keep_count = min(len(scored_moves), 10)
        
        # Always keep high-scoring moves
        top_score = scored_moves[0][1] if scored_moves else 0
        threshold = max(100, top_score * 0.1)  # Keep moves within 10% of top score or above 100
        
        pruned = []
        for move, score in scored_moves:
            if len(pruned) < keep_count or score >= threshold:
                pruned.append((move, score))
            else:
                break  # Stop adding once we hit limits
        
        return pruned
    
    # Helper methods (preserved from V8.0 but simplified)
    def new_game(self):
        """Reset search tables for a new game"""
        self.killer_moves.clear()
        self.history_scores.clear()
        self.evaluation_cache.clear()
        self.nodes_searched = 0
        
        # Reset performance stats
        self.search_stats = {
            'nodes_per_second': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
    
    def _store_killer_move(self, move: chess.Move, ply: int):
        """Store a killer move that caused a cutoff"""
        if ply not in self.killer_moves:
            self.killer_moves[ply] = []
        
        if move in self.killer_moves[ply]:
            self.killer_moves[ply].remove(move)
        
        self.killer_moves[ply].insert(0, move)
        
        if len(self.killer_moves[ply]) > 2:
            self.killer_moves[ply] = self.killer_moves[ply][:2]
    
    def _update_history_score(self, move: chess.Move, depth: int):
        """Update history heuristic score for a move"""
        move_key = f"{move.from_square}_{move.to_square}"
        bonus = depth * depth
        
        if move_key in self.history_scores:
            self.history_scores[move_key] += bonus
        else:
            self.history_scores[move_key] = bonus
        
        if self.history_scores[move_key] > 10000:
            self.history_scores[move_key] = 10000
    
    def _format_uci_score(self, score: float, search_depth: int) -> str:
        """Format score for UCI output - properly handle mate scores"""
        if abs(score) >= 28500:
            if score > 0:
                if score >= 29000:
                    depth_to_mate = 29000 - score + search_depth
                    mate_in = max(1, int((depth_to_mate + 1) / 2))
                else:
                    mate_in = max(1, int((29000 - score) / 2))
                return f"mate {mate_in}"
            else:
                if score <= -29000:
                    depth_to_mate = abs(score) - 29000 + search_depth
                    mate_in = max(1, int((depth_to_mate + 1) / 2))
                else:
                    mate_in = max(1, int((abs(score) - 29000) / 2))
                return f"mate -{mate_in}"
        else:
            return f"cp {int(score * 100)}"
    
    def _inject_opening_knowledge(self):
        """Inject basic opening knowledge into transposition table"""
        # TODO: Implement opening book or opening principles
        pass
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get current search statistics"""
        return self.search_stats.copy()


# Backward compatibility alias
V7P3REngineV81 = V7P3RCleanEngine
