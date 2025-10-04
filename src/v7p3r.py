#!/usr/bin/env python3
"""
V7P3R Chess Engine v12.5 - Intelligent Nudge System v2.0

Enhanced with iteratively reintroduced intelligent nudge system for improved opening play,
center control, and strategic positioning while maintaining performance optimization.

ARCHITECTURE:
- Phase 1: Core search (alpha-beta, TT, iterative deepening)
- Phase 2: Intelligent Nudge System v2.0 (1176+ analyzed positions)  
- Phase 3A: Advanced evaluation (pawns, king safety, castling)
- Clean codebase focused on playing strength and stability

VERSION LINEAGE:
- v10.8: Recovery baseline (19.5/30 tournament points)
- v11.x: Experimental variants (lessons learned, features extracted)
- v12.0: Clean evolution with proven improvements only
- v12.2: Performance optimized with tactical regression
- v12.4: Enhanced castling with balanced evaluation
- v12.5: Intelligent nudge system reintroduction

Author: Pat Snyder
"""

import time
import chess
import sys
import random
import json
import os
from typing import Optional, Tuple, List, Dict
from collections import defaultdict
from v7p3r_bitboard_evaluator import V7P3RScoringCalculationBitboard
from v7p3r_advanced_pawn_evaluator import V7P3RAdvancedPawnEvaluator
from v7p3r_king_safety_evaluator import V7P3RKingSafetyEvaluator


class PVTracker:
    """Tracks principal variation using predicted board states for instant move recognition"""
    
    def __init__(self):
        self.predicted_position_fen = None  # FEN of position we expect after opponent move
        self.next_our_move = None          # Our move to play if prediction hits
        self.remaining_pv_queue = []       # Remaining moves in PV [opp_move, our_move, opp_move, our_move, ...]
        self.following_pv = False          # Whether we're actively following PV
        self.pv_display_string = ""        # PV string for display purposes
        self.original_pv = []              # Store original PV for display
        
    def store_pv_from_search(self, starting_board: chess.Board, pv_moves: List[chess.Move]):
        """Store PV from search results, setting up for following"""
        self.original_pv = pv_moves.copy()  # Keep original for display
        
        if len(pv_moves) < 3:  # Need at least: our_move, opp_move, our_next_move
            self.clear()
            return
            
        # We're about to play pv_moves[0], so prepare for opponent response
        temp_board = starting_board.copy()
        temp_board.push(pv_moves[0])  # Make our first move
        
        if len(pv_moves) >= 2:
            # Predict position after opponent plays pv_moves[1]
            temp_board.push(pv_moves[1])  # Opponent's expected response
            self.predicted_position_fen = temp_board.fen()
            
            if len(pv_moves) >= 3:
                self.next_our_move = pv_moves[2]  # Our response to their response
                self.remaining_pv_queue = pv_moves[3:]  # Rest of PV
                self.pv_display_string = ' '.join(str(m) for m in pv_moves[2:])
                self.following_pv = True
            else:
                self.clear()
        else:
            self.clear()
    
    def clear(self):
        """Clear all PV following state"""
        self.predicted_position_fen = None
        self.next_our_move = None
        self.remaining_pv_queue = []
        self.following_pv = False
        self.pv_display_string = ""
        # Keep original_pv for display even after clearing following state
    
    def check_position_for_instant_move(self, current_board: chess.Board) -> Optional[chess.Move]:
        """Check if current position matches prediction - if so, return instant move"""
        if not self.following_pv or not self.predicted_position_fen:
            return None
            
        current_fen = current_board.fen()
        if current_fen == self.predicted_position_fen:
            # Position matches prediction - return instant move
            move_to_play = self.next_our_move
            
            # Clean UCI output for PV following - FIXED UCI format
            remaining_pv_str = self.pv_display_string if self.pv_display_string else str(move_to_play)
            print(f"info depth 1 score cp 0 nodes 0 time 0 pv {remaining_pv_str}")
            print(f"info string PV prediction match")
            
            # Set up for next prediction if we have more moves
            self._setup_next_prediction(current_board)
            
            return move_to_play
        else:
            # Position doesn't match - opponent broke PV, clear following
            self.clear()
            return None
    
    def _setup_next_prediction(self, current_board: chess.Board):
        """Set up prediction for next opponent move"""
        if len(self.remaining_pv_queue) < 2:  # Need at least opp_move, our_move
            self.clear()
            return
            
        # Make our move that we're about to play
        temp_board = current_board.copy()
        if self.next_our_move:  # Safety check
            temp_board.push(self.next_our_move)
        
        # Predict position after next opponent move
        next_opp_move = self.remaining_pv_queue[0]
        temp_board.push(next_opp_move)
        
        # Set up for next iteration
        self.predicted_position_fen = temp_board.fen()
        self.next_our_move = self.remaining_pv_queue[1] if len(self.remaining_pv_queue) >= 2 else None
        self.remaining_pv_queue = self.remaining_pv_queue[2:]  # Remove used moves
        self.pv_display_string = ' '.join(str(m) for m in [self.next_our_move] + self.remaining_pv_queue) if self.next_our_move else ""
        
        if not self.next_our_move:
            self.clear()  # No more moves to follow


class TranspositionEntry:
    """Entry in the transposition table"""
    def __init__(self, depth: int, score: int, best_move: Optional[chess.Move], 
                 node_type: str, zobrist_hash: int):
        self.depth = depth
        self.score = score
        self.best_move = best_move
        self.node_type = node_type  # 'exact', 'lowerbound', 'upperbound'
        self.zobrist_hash = zobrist_hash


class KillerMoves:
    """Killer move storage - 2 killer moves per depth"""
    def __init__(self):
        self.killers: Dict[int, List[chess.Move]] = defaultdict(list)
    
    def store_killer(self, move: chess.Move, depth: int):
        """Store a killer move at the given depth"""
        if move not in self.killers[depth]:
            self.killers[depth].insert(0, move)
            if len(self.killers[depth]) > 2:
                self.killers[depth].pop()
    
    def get_killers(self, depth: int) -> List[chess.Move]:
        """Get killer moves for the given depth"""
        return self.killers[depth]
    
    def is_killer(self, move: chess.Move, depth: int) -> bool:
        """Check if a move is a killer at the given depth"""
        return move in self.killers[depth]


class HistoryHeuristic:
    """History heuristic for move ordering"""
    def __init__(self):
        self.history: Dict[str, int] = defaultdict(int)
    
    def update_history(self, move: chess.Move, depth: int):
        """Update history score for a move"""
        move_key = f"{move.from_square}-{move.to_square}"
        self.history[move_key] += depth * depth
    
    def get_history_score(self, move: chess.Move) -> int:
        """Get history score for a move"""
        move_key = f"{move.from_square}-{move.to_square}"
        return self.history[move_key]


class ZobristHashing:
    """Zobrist hashing for transposition table"""
    def __init__(self):
        random.seed(12345)  # Deterministic for reproducibility
        self.piece_square_table = {}
        self.side_to_move = random.getrandbits(64)
        
        # Generate random numbers for each piece on each square
        for square in range(64):
            for piece_type in range(1, 7):  # PAWN to KING
                for color in [chess.WHITE, chess.BLACK]:
                    key = (square, piece_type, color)
                    self.piece_square_table[key] = random.getrandbits(64)
    
    def hash_position(self, board: chess.Board) -> int:
        """Generate Zobrist hash for the position"""
        hash_value = 0
        
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                key = (square, piece.piece_type, piece.color)
                hash_value ^= self.piece_square_table[key]
        
        if board.turn == chess.BLACK:
            hash_value ^= self.side_to_move
            
        return hash_value


class V7P3REngine:
    """V7P3R Chess Engine v12.5 - Intelligent Nudge System v2.0"""
    
    # V12.5 FEATURE TOGGLES - Intelligent Nudge System Configuration
    ENABLE_NUDGE_SYSTEM = True       # V12.5: Re-enabled with Intelligent Nudge System v2.0
    ENABLE_PV_FOLLOWING = True       # Keep - high value feature
    ENABLE_ADVANCED_EVALUATION = False  # V12.4: Disabled for performance testing
    
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
        
        # Evaluation components - V12.0: Proven stable (bitboard + advanced)
        self.bitboard_evaluator = V7P3RScoringCalculationBitboard(self.piece_values)
        self.advanced_pawn_evaluator = V7P3RAdvancedPawnEvaluator()
        self.king_safety_evaluator = V7P3RKingSafetyEvaluator()
        
        # Simple evaluation cache for speed
        self.evaluation_cache = {}  # position_hash -> evaluation
        
        # Advanced search infrastructure
        self.transposition_table: Dict[int, TranspositionEntry] = {}
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        self.zobrist = ZobristHashing()
        
        # V12.2 CONDITIONAL: Nudge System Integration (V12.5: Enhanced with Intelligent Nudges v2.0)
        if self.ENABLE_NUDGE_SYSTEM:
            # Initialize legacy nudge database
            self.nudge_database = {}
            self.nudge_stats = {
                'hits': 0,
                'misses': 0,
                'moves_boosted': 0,
                'positions_matched': 0,
                'instant_moves': 0,
                'instant_time_saved': 0.0
            }
            
            # V11 PHASE 2 ENHANCEMENT: Nudge threshold configuration
            self.nudge_instant_config = {
                'min_frequency': 8,        # Move must be played at least 8 times
                'min_eval': 0.4,          # Move must have eval improvement >= 0.4
                'confidence_threshold': 12.0  # Combined confidence score threshold
            }
            
            # V12.5: Initialize Intelligent Nudge System v2.0
            self.intelligent_nudges = None
            self._init_intelligent_nudges()
            
            self._load_nudge_database()
        else:
            # V12.2: Nudge system disabled - initialize empty placeholders
            self.nudge_database = {}
            self.nudge_stats = {'instant_time_saved': 0.0}  # Keep for compatibility
        
        # Configuration
        self.max_tt_entries = 50000  # Reasonable size for testing
        
        # Performance monitoring
        self.search_stats = {
            'nodes_per_second': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'tt_hits': 0,
            'tt_stores': 0,
            'killer_hits': 0,
            'nudge_hits': 0,        # V11 PHASE 2
            'nudge_positions': 0,   # V11 PHASE 2
        }
        
        # PV Following System - V10 OPTIMIZATION
        self.pv_tracker = PVTracker()
    
    def search(self, board: chess.Board, time_limit: float = 3.0, depth: Optional[int] = None, 
               alpha: float = -99999, beta: float = 99999, is_root: bool = True) -> chess.Move:
        """
        UNIFIED SEARCH - Single function with ALL advanced features:
        - Iterative deepening with stable best move handling (root level)
        - Alpha-beta pruning with negamax framework (recursive level)  
        - Transposition table with Zobrist hashing
        - Killer moves and history heuristic
        - Advanced move ordering with tactical detection
        - Proper time management with periodic checks
        - Full PV extraction and following
        - Quiescence search for tactical stability
        """
        
        # ROOT LEVEL: Iterative deepening with time management
        if is_root:
            self.nodes_searched = 0
            self.search_start_time = time.time()
            
            
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return chess.Move.null()

            # PV FOLLOWING OPTIMIZATION - check if current position triggers instant move
            if self.ENABLE_PV_FOLLOWING:
                pv_move = self.pv_tracker.check_position_for_instant_move(board)
                if pv_move:
                    return pv_move
            
            # V12.2 CONDITIONAL: Check for instant nudge moves (DISABLED for performance)
            if self.ENABLE_NUDGE_SYSTEM:
                instant_nudge_move = self._check_instant_nudge_move(board)
                if instant_nudge_move:
                    # Calculate time saved
                    time_saved = time_limit * 0.8  # Estimate time that would have been used
                    self.nudge_stats['instant_time_saved'] += time_saved

                    # Output instant move info - FIXED UCI format
                    print(f"info depth 1 score cp 50 nodes 0 time 0 pv {instant_nudge_move}")
                    print(f"info string Instant nudge move: {instant_nudge_move} (high confidence)")

                    return instant_nudge_move            # V11 ENHANCEMENT: Adaptive time management
            target_time, max_time = self._calculate_adaptive_time_allocation(board, time_limit)
            
            # Iterative deepening
            best_move = legal_moves[0]
            best_score = -99999
            
            for current_depth in range(1, self.default_depth + 1):
                iteration_start = time.time()
                
                # V11 ENHANCEMENT: Improved time checking with adaptive limits
                elapsed = time.time() - self.search_start_time
                if elapsed > target_time:
                    break
                
                # Predict if next iteration will exceed max time
                if current_depth > 1:
                    last_iteration_time = time.time() - iteration_start
                    if elapsed + (last_iteration_time * 2) > max_time:
                        break
                
                try:
                    # Store previous best in case iteration fails
                    previous_best = best_move
                    previous_score = best_score
                    
                    # Call recursive search for this depth
                    score, move = self._recursive_search(board, current_depth, -99999, 99999, time_limit)
                    
                    # Update best move if we got a valid result
                    if move and move != chess.Move.null():
                        best_move = move
                        best_score = score
                        
                        elapsed_ms = int((time.time() - self.search_start_time) * 1000)
                        nps = int(self.nodes_searched / max(elapsed_ms / 1000, 0.001))
                        
                        # Extract and display PV
                        pv_line = self._extract_pv(board, current_depth)
                        pv_string = " ".join([str(m) for m in pv_line])
                        
                        # Store PV for following optimization
                        if current_depth >= 4 and len(pv_line) >= 3:
                            self.pv_tracker.store_pv_from_search(board, pv_line)
                        
                        print(f"info depth {current_depth} score cp {int(score)} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {pv_string}")
                        sys.stdout.flush()
                    else:
                        # Restore previous best if iteration failed
                        best_move = previous_best
                        best_score = previous_score
                    
                    # V11 ENHANCEMENT: Better time management for next iteration
                    elapsed = time.time() - self.search_start_time
                    iteration_time = time.time() - iteration_start
                    
                    if elapsed > target_time:
                        break
                    elif elapsed + (iteration_time * 1.5) > max_time:
                        break  # Don't start next iteration if likely to exceed max time
                        
                except Exception as e:
                    print(f"info string Search interrupted at depth {current_depth}: {e}")
                    break
                    
            return best_move
        
        # This should never be called directly with is_root=False from external code
        else:
            # Fallback - call the recursive search method
            score, move = self._recursive_search(board, depth or 1, alpha, beta, time_limit)
            return move if move else chess.Move.null()
    
    def _recursive_search(self, board: chess.Board, search_depth: int, alpha: float, beta: float, time_limit: float) -> Tuple[float, Optional[chess.Move]]:
        """
        Recursive alpha-beta search with all advanced features
        Returns (score, best_move) tuple
        """
        self.nodes_searched += 1
        
        # CRITICAL: Time checking during recursive search to prevent timeouts
        if hasattr(self, 'search_start_time') and self.nodes_searched % 1000 == 0:
            elapsed = time.time() - self.search_start_time
            if elapsed > time_limit:
                # Emergency return with current best evaluation
                return self._evaluate_position(board), None
        
        # 1. TRANSPOSITION TABLE PROBE
        tt_hit, tt_score, tt_move = self._probe_transposition_table(board, search_depth, int(alpha), int(beta))
        if tt_hit:
            return float(tt_score), tt_move
        
        # 2. TERMINAL CONDITIONS
        if search_depth == 0:
            # Enter quiescence search for tactical stability
            score = self._quiescence_search(board, alpha, beta, 4)
            return score, None
            
        if board.is_game_over():
            if board.is_checkmate():
                score = -29000.0 + (self.default_depth - search_depth)  # Prefer quicker mates
            else:
                score = 0.0  # Stalemate
            return score, None
        
        # 3. NULL MOVE PRUNING
        if (search_depth >= 3 and not board.is_check() and 
            self._has_non_pawn_pieces(board) and beta - alpha > 1):
            
            board.turn = not board.turn
            null_score, _ = self._recursive_search(board, search_depth - 2, -beta, -beta + 1, time_limit)
            null_score = -null_score
            board.turn = not board.turn
            
            if null_score >= beta:
                return null_score, None
        
        # 4. MOVE GENERATION AND ORDERING
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0, None
        
        ordered_moves = self._order_moves_advanced(board, legal_moves, search_depth, tt_move)
        
        # 5. MAIN SEARCH LOOP (NEGAMAX WITH ALPHA-BETA)
        best_score = -99999.0
        best_move = None
        original_alpha = alpha
        moves_searched = 0
        
        for move in ordered_moves:
            board.push(move)
            
            # V11 ENHANCEMENT: Enhanced Late Move Reduction
            reduction = self._calculate_lmr_reduction(move, moves_searched, search_depth, board)
            
            # Search with possible reduction
            if reduction > 0:
                score, _ = self._recursive_search(board, search_depth - 1 - reduction, -beta, -alpha, time_limit)
                score = -score
                
                # Re-search at full depth if reduced search failed high
                if score > alpha:
                    score, _ = self._recursive_search(board, search_depth - 1, -beta, -alpha, time_limit)
                    score = -score
            else:
                score, _ = self._recursive_search(board, search_depth - 1, -beta, -alpha, time_limit)
                score = -score
            
            board.pop()
            moves_searched += 1
            
            # Update best move
            if best_move is None or score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            if alpha >= beta:
                # Beta cutoff - update heuristics
                if not board.is_capture(move):
                    self.killer_moves.store_killer(move, search_depth)
                    self.history_heuristic.update_history(move, search_depth)
                    self.search_stats['killer_hits'] += 1
                break
        
        # 7. TRANSPOSITION TABLE STORE
        self._store_transposition_table(board, search_depth, int(best_score), best_move, int(original_alpha), int(beta))
        
        return best_score, best_move
    
    def _order_moves_advanced(self, board: chess.Board, moves: List[chess.Move], depth: int, 
                              tt_move: Optional[chess.Move] = None) -> List[chess.Move]:
        """V11 PHASE 2 ENHANCED move ordering - TT, NUDGES, MVV-LVA, Checks, Killers, BITBOARD TACTICS"""
        if len(moves) <= 2:
            return moves
        
        # Pre-calculate move categories for efficiency
        captures = []
        checks = []
        killers = []
        quiet_moves = []
        tactical_moves = []  # NEW: Bitboard tactical moves
        nudge_moves = []     # V11 PHASE 2: Nudge system moves
        tt_moves = []
        
        killer_set = set(self.killer_moves.get_killers(depth))
        
        # Check if current position has nudge data
        position_has_nudges = self._get_position_key(board) in self.nudge_database
        if position_has_nudges:
            self.nudge_stats['positions_matched'] += 1
        
        for move in moves:
            # V12.2 CONDITIONAL: Calculate nudge bonus for this move (DISABLED for performance)
            if self.ENABLE_NUDGE_SYSTEM:
                nudge_bonus = self._get_nudge_bonus(board, move)
            else:
                nudge_bonus = 0.0  # No nudge bonus when disabled
            
            # 1. Transposition table move (highest priority)
            if tt_move and move == tt_move:
                tt_moves.append(move)
            
            # 2. Nudge moves (second highest priority - V11 PHASE 2, disabled in v12.2)
            elif nudge_bonus > 0:
                nudge_moves.append((nudge_bonus, move))
            
            # 3. Captures (will be sorted by MVV-LVA)
            elif board.is_capture(move):
                victim = board.piece_at(move.to_square)
                victim_value = self.piece_values.get(victim.piece_type, 0) if victim else 0
                attacker = board.piece_at(move.from_square)
                attacker_value = self.piece_values.get(attacker.piece_type, 0) if attacker else 0
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                mvv_lva_score = victim_value * 100 - attacker_value
                
                # Add tactical bonus using bitboards
                tactical_bonus = self._detect_bitboard_tactics(board, move)
                total_score = mvv_lva_score + tactical_bonus
                
                captures.append((total_score, move))
            
            # 4. Checks (high priority for tactical play)
            elif board.gives_check(move):
                # Add tactical bonus for checking moves too
                tactical_bonus = self._detect_bitboard_tactics(board, move)
                checks.append((tactical_bonus, move))
            
            # 5. Killer moves
            elif move in killer_set:
                killers.append(move)
                self.search_stats['killer_hits'] += 1
            
            # 6. Check for tactical patterns in quiet moves
            else:
                history_score = self.history_heuristic.get_history_score(move)
                tactical_bonus = self._detect_bitboard_tactics(board, move)
                
                if tactical_bonus > 20.0:  # Significant tactical move
                    tactical_moves.append((tactical_bonus + history_score, move))
                else:
                    quiet_moves.append((history_score, move))
        
        # Sort all move categories by their scores
        captures.sort(key=lambda x: x[0], reverse=True)
        checks.sort(key=lambda x: x[0], reverse=True)
        tactical_moves.sort(key=lambda x: x[0], reverse=True)
        quiet_moves.sort(key=lambda x: x[0], reverse=True)
        nudge_moves.sort(key=lambda x: x[0], reverse=True)  # V11 PHASE 2
        
        # Combine in V11 PHASE 2 ENHANCED order
        ordered = []
        ordered.extend(tt_moves)  # TT move first
        ordered.extend([move for _, move in nudge_moves])  # Then nudge moves (V11 PHASE 2)
        ordered.extend([move for _, move in captures])  # Then captures (with tactical bonus)
        ordered.extend([move for _, move in checks])  # Then checks (with tactical bonus)
        ordered.extend([move for _, move in tactical_moves])  # Then tactical patterns
        ordered.extend(killers)  # Then killers
        ordered.extend([move for _, move in quiet_moves])  # Then quiet moves
        
        return ordered
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """V12.2 OPTIMIZED: Position evaluation with Zobrist-based caching"""
        # V12.2 OPTIMIZATION: Use Zobrist hash instead of expensive FEN for cache key
        cache_key = self.zobrist.hash_position(board)
        
        if cache_key in self.evaluation_cache:
            self.search_stats['cache_hits'] += 1
            return self.evaluation_cache[cache_key]
        
        self.search_stats['cache_misses'] += 1
        
        # V10 Base bitboard evaluation for material and basic positioning
        white_base = self.bitboard_evaluator.calculate_score_optimized(board, True)
        black_base = self.bitboard_evaluator.calculate_score_optimized(board, False)
        
        # V12.2 CONDITIONAL: Advanced evaluation components 
        try:
            if self.ENABLE_ADVANCED_EVALUATION:
                # V11 PHASE 3A: Advanced pawn structure evaluation
                white_pawn_score = self.advanced_pawn_evaluator.evaluate_pawn_structure(board, True)
                black_pawn_score = self.advanced_pawn_evaluator.evaluate_pawn_structure(board, False)
                
                # Advanced king safety evaluation
                white_king_score = self.king_safety_evaluator.evaluate_king_safety(board, True)
                black_king_score = self.king_safety_evaluator.evaluate_king_safety(board, False)
            else:
                # V12.2: Simplified evaluation for performance
                white_pawn_score = 0  # Skip advanced pawn evaluation
                black_pawn_score = 0
                
                # V12.2: Simple king safety - just check if king is castled
                white_king_score = self._simple_king_safety(board, True)
                black_king_score = self._simple_king_safety(board, False)
            
            # V10.6 ROLLBACK: Tactical pattern evaluation disabled for performance
            # Phase 3B showed 70% performance degradation in tournament play
            white_tactical_score = 0  # V10.6: Disabled Phase 3B
            black_tactical_score = 0  # V10.6: Disabled Phase 3B
            
            # Combine all evaluation components
            white_total = white_base + white_pawn_score + white_king_score + white_tactical_score
            black_total = black_base + black_pawn_score + black_king_score + black_tactical_score
            
        except Exception as e:
            # Fallback to base evaluation if advanced evaluation fails
            white_total = white_base
            black_total = black_base
        
        # Calculate final score from current player's perspective
        if board.turn:  # White to move
            final_score = white_total - black_total
        else:  # Black to move
            final_score = black_total - white_total
        
        # Cache the result
        self.evaluation_cache[cache_key] = final_score
        return final_score
    
    def _simple_king_safety(self, board: chess.Board, color: bool) -> float:
        """V12.2: Simplified king safety evaluation for performance"""
        score = 0.0
        
        # Basic castling bonus
        if color == chess.WHITE:
            if board.has_kingside_castling_rights(color):
                score += 15
            if board.has_queenside_castling_rights(color):
                score += 10
        else:
            if board.has_kingside_castling_rights(color):
                score += 15
            if board.has_queenside_castling_rights(color):
                score += 10
        
        return score
    
    def _probe_transposition_table(self, board: chess.Board, depth: int, alpha: int, beta: int) -> Tuple[bool, int, Optional[chess.Move]]:
        """Probe transposition table for this position - PHASE 1 FEATURE"""
        zobrist_hash = self.zobrist.hash_position(board)
        
        if zobrist_hash in self.transposition_table:
            entry = self.transposition_table[zobrist_hash]
            
            # Only use if searched to sufficient depth
            if entry.depth >= depth:
                self.search_stats['tt_hits'] += 1
                
                # Check if we can use the score
                if entry.node_type == 'exact':
                    return True, entry.score, entry.best_move
                elif entry.node_type == 'lowerbound' and entry.score >= beta:
                    return True, entry.score, entry.best_move
                elif entry.node_type == 'upperbound' and entry.score <= alpha:
                    return True, entry.score, entry.best_move
        
        return False, 0, None
    
    def _store_transposition_table(self, board: chess.Board, depth: int, score: int, 
                                   best_move: Optional[chess.Move], alpha: int, beta: int):
        """Store result in transposition table - PHASE 1 FEATURE"""
        zobrist_hash = self.zobrist.hash_position(board)
        
        # Determine node type
        if score <= alpha:
            node_type = 'upperbound'
        elif score >= beta:
            node_type = 'lowerbound'
        else:
            node_type = 'exact'
        
        # Simple replacement strategy for performance
        if len(self.transposition_table) >= self.max_tt_entries:
            # Clear 25% of entries when full (simple aging)
            entries = list(self.transposition_table.items())
            entries.sort(key=lambda x: x[1].depth, reverse=True)
            self.transposition_table = dict(entries[:int(self.max_tt_entries * 0.75)])
        
        entry = TranspositionEntry(depth, score, best_move, node_type, zobrist_hash)
        self.transposition_table[zobrist_hash] = entry
        self.search_stats['tt_stores'] += 1
    
    def _has_non_pawn_pieces(self, board: chess.Board) -> bool:
        """Check if the current side has non-pawn pieces (for null move pruning)"""
        current_color = board.turn
        for square in range(64):
            piece = board.piece_at(square)
            if piece and piece.color == current_color and piece.piece_type != chess.PAWN:
                return True
        return False
    
    def _extract_pv(self, board: chess.Board, max_depth: int) -> List[chess.Move]:
        """Extract principal variation from transposition table"""
        pv = []
        temp_board = board.copy()
        
        for depth in range(max_depth, 0, -1):
            zobrist_hash = self.zobrist.hash_position(temp_board)
            
            if zobrist_hash in self.transposition_table:
                entry = self.transposition_table[zobrist_hash]
                if entry.best_move and entry.best_move in temp_board.legal_moves:
                    pv.append(entry.best_move)
                    temp_board.push(entry.best_move)
                else:
                    break
            else:
                break
        
        return pv
    
    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int) -> float:
        """
        Quiescence search for tactical stability - V10 PHASE 2
        Only search captures and checks to avoid horizon effects
        """
        self.nodes_searched += 1
        
        # Stand pat evaluation
        stand_pat = self._evaluate_position(board)
        
        # Beta cutoff on stand pat
        if stand_pat >= beta:
            return beta
        
        # Update alpha if stand pat is better
        if stand_pat > alpha:
            alpha = stand_pat
        
        # Depth limit reached
        if depth <= 0:
            return stand_pat
        
        # Generate and search tactical moves only
        legal_moves = list(board.legal_moves)
        tactical_moves = []
        
        for move in legal_moves:
            # Only consider captures and checks for quiescence
            if board.is_capture(move) or board.gives_check(move):
                tactical_moves.append(move)
        
        # If no tactical moves, return stand pat
        if not tactical_moves:
            return stand_pat
        
        # Sort tactical moves by MVV-LVA for better ordering
        capture_scores = []
        for move in tactical_moves:
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                victim_value = self.piece_values.get(victim.piece_type, 0) if victim else 0
                attacker = board.piece_at(move.from_square)
                attacker_value = self.piece_values.get(attacker.piece_type, 0) if attacker else 0
                mvv_lva = victim_value * 100 - attacker_value
                capture_scores.append((mvv_lva, move))
            else:
                # Check moves get lower priority
                capture_scores.append((0, move))
        
        # Sort by MVV-LVA score
        capture_scores.sort(key=lambda x: x[0], reverse=True)
        ordered_tactical = [move for _, move in capture_scores]
        
        # Search tactical moves
        best_score = stand_pat
        for move in ordered_tactical:
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha, depth - 1)
            board.pop()
            
            if score > best_score:
                best_score = score
            
            if score > alpha:
                alpha = score
            
            if alpha >= beta:
                break  # Beta cutoff
        
        return best_score
    
    def _detect_bitboard_tactics(self, board: chess.Board, move: chess.Move) -> float:
        """
        V11 PHASE 3B ENHANCED: Detect tactical patterns using advanced pattern detector
        Returns a bonus score for tactical moves (pins, forks, skewers, discovered attacks)
        """
        tactical_bonus = 0.0
        
        # Make the move to analyze the resulting position
        board.push(move)
        
        try:
            our_color = not board.turn  # We just moved, so it's opponent's turn
            
            # V10.6 ROLLBACK: Use legacy tactical analysis only  
            # Advanced tactical patterns disabled due to 70% performance degradation
            tactical_bonus += 0  # V10.6: Disabled Phase 3B advanced tactical detection
            
            # Legacy bitboard tactics for additional analysis
            moving_piece = board.piece_at(move.to_square)
            if moving_piece:
                fork_bonus = self._analyze_fork_bitboard(board, move.to_square, moving_piece, board.turn)
                tactical_bonus += fork_bonus
                
                # Analyze for pins and skewers using ray attacks
                if moving_piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    pin_skewer_bonus = self._analyze_pins_skewers_bitboard(board, move.to_square, moving_piece, board.turn)
                    tactical_bonus += pin_skewer_bonus
            
        except Exception:
            # If analysis fails, return 0 bonus
            pass
        finally:
            board.pop()
        
        return tactical_bonus
    
    def _analyze_fork_bitboard(self, board: chess.Board, square: int, piece: chess.Piece, enemy_color: chess.Color) -> float:
        """Analyze fork patterns using bitboards"""
        if piece.piece_type == chess.KNIGHT:
            # Knight fork detection
            attacks = self.bitboard_evaluator.bitboard_evaluator.KNIGHT_ATTACKS[square]
            enemy_pieces = 0
            high_value_targets = 0
            
            for target_sq in range(64):
                if attacks & (1 << target_sq):
                    target_piece = board.piece_at(target_sq)
                    if target_piece and target_piece.color == enemy_color:
                        enemy_pieces += 1
                        if target_piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                            high_value_targets += 1
            
            # Knight forking 2+ pieces gets bonus, more for high-value targets
            if enemy_pieces >= 2:
                return 50.0 + (high_value_targets * 25.0)
        
        return 0.0
    
    def _analyze_pins_skewers_bitboard(self, board: chess.Board, square: int, piece: chess.Piece, enemy_color: chess.Color) -> float:
        """Analyze pin and skewer patterns using ray attacks"""
        # This is a simplified version - full implementation would need sliding piece attack generation
        # For now, just give a small bonus for pieces that could create pins/skewers
        
        if piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            # Look for aligned enemy pieces that could be pinned/skewered
            bonus = 0.0
            
            # Check if we're attacking towards the enemy king
            enemy_king_sq = None
            for sq in range(64):
                p = board.piece_at(sq)
                if p and p.piece_type == chess.KING and p.color == enemy_color:
                    enemy_king_sq = sq
                    break
            
            if enemy_king_sq is not None:
                # Simple heuristic: if we're on the same rank/file/diagonal as enemy king
                sq_rank, sq_file = divmod(square, 8)
                king_rank, king_file = divmod(enemy_king_sq, 8)
                
                if (sq_rank == king_rank or sq_file == king_file or 
                    abs(sq_rank - king_rank) == abs(sq_file - king_file)):
                    bonus += 15.0  # Potential pin/skewer bonus
            
            return bonus
        
        return 0.0
    
    def notify_move_played(self, move: chess.Move, board_before_move: chess.Board):
        """Notify engine that a move was played (for PV following)
        
        Args:
            move: The move that was played
            board_before_move: The board position BEFORE the move was made
        """
        # We don't need this method anymore - position checking is automatic
        # The new approach checks positions directly when search() is called
        pass

    def new_game(self):
        """Reset for a new game"""
        self.evaluation_cache.clear()
        self.transposition_table.clear()
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        self.nodes_searched = 0
        
        # Clear PV following data
        self.pv_tracker.clear()
        
        # Reset stats
        for key in self.search_stats:
            self.search_stats[key] = 0

    @property
    def principal_variation(self) -> List[chess.Move]:
        """Get the current principal variation"""
        # Return the original PV from the last search for display purposes
        return self.pv_tracker.original_pv.copy() if self.pv_tracker.original_pv else []

    def perft(self, board: chess.Board, depth: int, divide: bool = False, root_call: bool = True) -> int:
        """
        V11 ENHANCEMENT: Perft (Performance Test) - counts nodes at given depth
        
        This is essential for move generation validation and testing.
        Standard chess engine requirement that was missing in V10.2.
        
        Args:
            board: Current chess position
            depth: Depth to search (number of plies)
            divide: If True, show per-move breakdown at root
            root_call: Internal flag for root level tracking
            
        Returns:
            Total number of leaf nodes at specified depth
        """
        if depth == 0:
            return 1
        
        if root_call:
            self.perft_start_time = time.time()
            self.perft_nodes = 0
        
        legal_moves = list(board.legal_moves)
        total_nodes = 0
        
        if divide and depth == 1:
            # For divide, show each move's contribution
            for move in legal_moves:
                board.push(move)
                nodes = 1  # At depth 1, each legal move contributes 1 node
                board.pop()
                total_nodes += nodes
                if root_call:
                    print(f"{move}: {nodes}")
        else:
            # Normal perft counting
            for move in legal_moves:
                board.push(move)
                nodes = self.perft(board, depth - 1, False, False)
                board.pop()
                total_nodes += nodes
                
                if divide and root_call:
                    print(f"{move}: {nodes}")
        
        if root_call:
            elapsed = time.time() - self.perft_start_time
            nps = int(total_nodes / max(elapsed, 0.001))
            print(f"\nNodes searched: {total_nodes}")
            print(f"Time: {elapsed:.3f}s")
            print(f"Nodes per second: {nps}")
        
        return total_nodes

    def _calculate_adaptive_time_allocation(self, board: chess.Board, base_time_limit: float) -> Tuple[float, float]:
        """
        V11 ENHANCEMENT: Adaptive time management based on position complexity
        
        Returns: (target_time, max_time)
        """
        moves_played = len(board.move_stack)
        legal_moves = list(board.legal_moves)
        num_legal_moves = len(legal_moves)
        
        # Base time factor
        time_factor = 1.0
        
        # Game phase adjustment
        if moves_played < 15:  # Opening
            time_factor *= 0.8  # Faster in opening
        elif moves_played < 40:  # Middle game
            time_factor *= 1.2  # More time in complex middle game
        else:  # Endgame
            time_factor *= 0.9  # Moderate time in endgame
        
        # Position complexity factors
        if board.is_check():
            time_factor *= 1.3  # More time when in check
        
        if num_legal_moves <= 5:
            time_factor *= 0.7  # Less time with few options
        elif num_legal_moves >= 35:
            time_factor *= 1.4  # More time with many options
        
        # Material balance consideration
        our_material = self._count_material(board, board.turn)
        their_material = self._count_material(board, not board.turn)
        material_diff = our_material - their_material
        
        if material_diff < -300:  # We're behind
            time_factor *= 1.2  # Take more time when behind
        elif material_diff > 300:  # We're ahead
            time_factor *= 0.9  # Play faster when ahead
        
        # Calculate final times
        target_time = min(base_time_limit * time_factor * 0.8, base_time_limit * 0.9)
        max_time = min(base_time_limit * time_factor, base_time_limit)
        
        return target_time, max_time
    
    def _count_material(self, board: chess.Board, color: bool) -> int:
        """Count total material for a color"""
        total = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            pieces = board.pieces(piece_type, color)
            total += len(pieces) * self.piece_values[piece_type]
        return total

    def _calculate_lmr_reduction(self, move: chess.Move, moves_searched: int, search_depth: int, board: chess.Board) -> int:
        """
        V11 ENHANCEMENT: Calculate Late Move Reduction amount based on move characteristics
        
        Returns reduction amount (0 = no reduction, 1+ = reduction plies)
        """
        # No reduction for first few moves
        if moves_searched < 3:
            return 0
        
        # No reduction at low depths
        if search_depth < 3:
            return 0
        
        # No reduction for tactical moves
        if (board.is_capture(move) or board.is_check() or 
            self.killer_moves.is_killer(move, search_depth)):
            return 0
        
        # Calculate base reduction
        reduction = 1
        
        # Increase reduction for later moves
        if moves_searched >= 8:
            reduction += 1
        if moves_searched >= 16:
            reduction += 1
        
        # Increase reduction at higher depths
        if search_depth >= 6:
            reduction += 1
        
        # Reduce reduction for good history moves
        history_score = self.history_heuristic.get_history_score(move)
        if history_score > 50:  # High history score
            reduction = max(0, reduction - 1)
        
        # Maximum reduction cap
        reduction = min(reduction, search_depth - 1)
        
        return reduction

    # V12.2 CONDITIONAL: NUDGE SYSTEM METHODS (disabled for performance)
    
    def _load_nudge_database(self):
        """V12.2: Load nudge database only if enabled"""
        if not self.ENABLE_NUDGE_SYSTEM:
            print("info string V12.2: Nudge system disabled for performance")
            return
            
        """Load the enhanced nudge database from JSON file - V12.0 upgrade with PyInstaller support"""
        try:
            # V12.0: Handle both development and PyInstaller bundled execution
            if getattr(sys, 'frozen', False):
                # Running as PyInstaller executable
                bundle_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
                enhanced_path = os.path.join(bundle_dir, 'v7p3r_enhanced_nudges.json')
                basic_path = os.path.join(bundle_dir, 'v7p3r_nudge_database.json')
            else:
                # Running as Python script
                current_dir = os.path.dirname(os.path.abspath(__file__))
                enhanced_path = os.path.join(current_dir, 'v7p3r_enhanced_nudges.json')
                basic_path = os.path.join(current_dir, 'v7p3r_nudge_database.json')
            
            if os.path.exists(enhanced_path):
                with open(enhanced_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'positions' in data:
                        self.nudge_database = data['positions']
                        tactical_count = sum(1 for pos in data['positions'].values() 
                                           if pos.get('is_tactical', False))
                    else:
                        # Old format
                        self.nudge_database = data
            elif os.path.exists(basic_path):
                with open(basic_path, 'r', encoding='utf-8') as f:
                    self.nudge_database = json.load(f)
            else:
                self.nudge_database = {}
        except Exception as e:
            print(f"info string Error loading nudge database: {e}")
            self.nudge_database = {}
    
    def _get_position_key(self, board: chess.Board) -> str:
        """Generate position key for nudge database lookup"""
        # Use FEN without halfmove and fullmove clocks for broader matching
        fen_parts = board.fen().split(' ')
        if len(fen_parts) >= 4:
            # Keep: position, turn, castling, en passant
            key_fen = ' '.join(fen_parts[:4])
        else:
            key_fen = board.fen()
        
        # Generate hash key similar to nudge database format
        import hashlib
        return hashlib.md5(key_fen.encode()).hexdigest()[:12]
    
    def _get_nudge_bonus(self, board: chess.Board, move: chess.Move) -> float:
        """V12.5: Enhanced nudge bonus using Intelligent Nudge System v2.0"""
        if not self.ENABLE_NUDGE_SYSTEM:
            return 0.0
        
        total_bonus = 0.0
        move_uci = move.uci()
        
        # 1. INTELLIGENT NUDGE SYSTEM v2.0 - Performance optimized
        if hasattr(self, 'intelligent_nudges') and self.intelligent_nudges is not None:
            # Opening move bonuses for better center control
            if board.fullmove_number <= 8:
                opening_bonus = self.intelligent_nudges.get_opening_bonus(move_uci, board.fullmove_number, board.fen())
                total_bonus += opening_bonus
            
            # Move ordering bonuses for historically good moves
            ordering_bonus = self.intelligent_nudges.get_move_ordering_bonus(move_uci)
            total_bonus += ordering_bonus
            
            # Check if this is a preferred move with confidence
            is_preferred, confidence_bonus = self.intelligent_nudges.is_preferred_move(move_uci, board.fen())
            if is_preferred:
                total_bonus += confidence_bonus
        
        # 2. LEGACY NUDGE DATABASE (if available)
        try:
            position_key = self._get_position_key(board)
            
            # Check if position exists in nudge database
            if position_key in self.nudge_database:
                position_data = self.nudge_database[position_key]
                
                # Check if move exists in nudge data
                if 'moves' in position_data and move_uci in position_data['moves']:
                    move_data = position_data['moves'][move_uci]
                    
                    # Calculate bonus based on frequency and evaluation
                    frequency = move_data.get('frequency', 1)
                    evaluation = move_data.get('eval', 0.0)
                    confidence = move_data.get('confidence', 0.5)
                    
                    # Legacy bonus calculation (reduced to avoid double-counting)
                    if confidence >= 0.5:  # Only trust high-confidence moves
                        legacy_bonus = min(30.0, frequency * 5 + max(0, evaluation) * 10)
                        total_bonus += legacy_bonus
                    
                    # Update statistics
                    self.nudge_stats['hits'] += 1
                    self.nudge_stats['moves_boosted'] += 1
                else:
                    self.nudge_stats['misses'] += 1
            
        except Exception as e:
            # Silently handle errors to avoid disrupting search
            pass
        
        # Scale total bonus to prevent overwhelming other move ordering factors
        scaled_bonus = min(25.0, total_bonus * 0.4)  # Max 25 points, 40% influence
        
        return scaled_bonus

    def _check_instant_nudge_move(self, board: chess.Board) -> Optional[chess.Move]:
        """V12.2: Check for instant nudge moves (disabled)"""
        if not self.ENABLE_NUDGE_SYSTEM:
            return None
            
        """
        V11 PHASE 2 ENHANCEMENT: Check for instant nudge moves that bypass search
        Returns move if confidence is high enough, None otherwise
        """
        try:
            position_key = self._get_position_key(board)
            
            # Check if position exists in nudge database
            if position_key not in self.nudge_database:
                return None
            
            position_data = self.nudge_database[position_key]
            moves_data = position_data.get('moves', {})
            
            if not moves_data:
                return None
            
            best_move = None
            best_confidence = 0.0
            
            # Evaluate all nudge moves for instant play criteria
            for move_uci, move_data in moves_data.items():
                try:
                    move = chess.Move.from_uci(move_uci)
                    
                    # Verify move is legal
                    if move not in board.legal_moves:
                        continue
                    
                    frequency = move_data.get('frequency', 0)
                    evaluation = move_data.get('eval', 0.0)
                    
                    # Check minimum thresholds
                    if (frequency < self.nudge_instant_config['min_frequency'] or 
                        evaluation < self.nudge_instant_config['min_eval']):
                        continue
                    
                    # Calculate confidence score (frequency + eval bonus)
                    confidence = frequency + (evaluation * 10)  # Scale eval to match frequency range
                    
                    # Track best candidate
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_move = move
                
                except:
                    continue
            
            # Check if best move meets confidence threshold
            if (best_move and 
                best_confidence >= self.nudge_instant_config['confidence_threshold']):
                
                # Update statistics
                self.nudge_stats['instant_moves'] += 1
                
                return best_move
            
            return None
            
        except Exception as e:
            # Silently handle errors to avoid disrupting search
            return None
    
    def _init_intelligent_nudges(self):
        """V12.5: Initialize the Intelligent Nudge System v2.0"""
        try:
            from .v7p3r_intelligent_nudges import V7P3RIntelligentNudges
            self.intelligent_nudges = V7P3RIntelligentNudges()
        except (ImportError, ModuleNotFoundError):
            # Try without relative import
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(__file__))
                from v7p3r_intelligent_nudges import V7P3RIntelligentNudges
                self.intelligent_nudges = V7P3RIntelligentNudges()
            except (ImportError, ModuleNotFoundError):
                self.intelligent_nudges = None
        except Exception as e:
            self.intelligent_nudges = None
