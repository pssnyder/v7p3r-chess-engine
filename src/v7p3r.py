#!/usr/bin/env python3
"""
V7P3R Chess Engine v11.3 - Built on V10.6 Proven Baseline
Phase 1: Core search + Phase 2: Nudge system + Phase 3A: Advanced evaluation
Phase 3B: Tactical patterns (NEW - implementing with acceptance criteria)
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
from v7p3r_tactical_pattern_detector import V7P3RTacticalPatternDetector  # V11.3 RE-ENABLED PHASE 3B

# V11.5 PERFORMANCE FIX: Tactical cache for eliminating redundant pattern detection
class TacticalCache:
    """High-speed tactical result cache"""
    def __init__(self, max_size: int = 5000):
        self.cache = {}  # FEN -> (score, timestamp)
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
    def get_cached_result(self, board, color):
        fen = board.fen()
        if fen in self.cache:
            self.hits += 1
            score, timestamp = self.cache[fen]
            return score if color == board.turn else -score
        self.misses += 1
        return None
    
    def cache_result(self, board, score, color):
        fen = board.fen()
        # Store from white's perspective
        if color != chess.WHITE:
            score = -score
        self.cache[fen] = (score, time.time())
        if len(self.cache) > self.max_size:
            # Remove oldest 25%
            sorted_entries = sorted(self.cache.items(), key=lambda x: x[1][1])
            prune_count = len(sorted_entries) // 4
            for i in range(prune_count):
                del self.cache[sorted_entries[i][0]]
    
    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        
    def get_stats(self):
        """Get cache performance statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate_percent': hit_rate,
            'cache_size': len(self.cache),
            'enabled': True
        }



# V11.3 QUIESCENCE OPTIMIZATION CONSTANTS
QUIESCENCE_OPTIMIZATION = True
MAX_QUIESCENCE_DEPTH = 4  # Limit quiescence depth
MIN_CAPTURE_VALUE = 100   # Only consider captures worth >= 1 pawn

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
            
            # Clean UCI output for PV following
            remaining_pv_str = self.pv_display_string if self.pv_display_string else str(move_to_play)
            print(f"info depth PV score cp 0 nodes 0 time 0 pv {remaining_pv_str}")
            
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
        
        # V11.5 PERFORMANCE: Position hash cache for repeated calculations
        self._position_hash_cache = {}
        self._position_hash_cache_size = 1000  # Limit cache size
        
        # Generate random numbers for each piece on each square
        for square in range(64):
            for piece_type in range(1, 7):  # PAWN to KING
                for color in [chess.WHITE, chess.BLACK]:
                    key = (square, piece_type, color)
                    self.piece_square_table[key] = random.getrandbits(64)
    
    def hash_position(self, board: chess.Board) -> int:
        """Generate Zobrist hash for the position - V11.5 OPTIMIZED with caching"""
        # V11.5 PERFORMANCE BOOST: Use cached hash if available
        board_fen = board.fen()
        if board_fen in self._position_hash_cache:
            return self._position_hash_cache[board_fen]
        
        hash_value = 0
        
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                key = (square, piece.piece_type, piece.color)
                hash_value ^= self.piece_square_table[key]
        
        if board.turn == chess.BLACK:
            hash_value ^= self.side_to_move
        
        # V11.5 PERFORMANCE: Cache the result, with size limit
        if len(self._position_hash_cache) < self._position_hash_cache_size:
            self._position_hash_cache[board_fen] = hash_value
        
        return hash_value


class V7P3REngine:
    """V7P3R Chess Engine v9.5 - Bitboard-powered version"""
    
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
        
        # Evaluation components - V10 BITBOARD POWERED + V11 PHASE 3A ADVANCED + V11.3 TACTICAL RE-ENABLED
        self.bitboard_evaluator = V7P3RScoringCalculationBitboard(self.piece_values)
        self.advanced_pawn_evaluator = V7P3RAdvancedPawnEvaluator()  # V11 PHASE 3A
        self.king_safety_evaluator = V7P3RKingSafetyEvaluator()      # V11 PHASE 3A
        self.tactical_pattern_detector = V7P3RTacticalPatternDetector()  # V11.3 RE-ENABLED PHASE 3B
        
        # V11.5 PERFORMANCE FIX: Tactical cache to eliminate redundant pattern detection
        self.tactical_cache = TacticalCache(max_size=5000)
        
        # Simple evaluation cache for speed
        self.evaluation_cache = {}  # position_hash -> evaluation
        
        # V11.3 PERFORMANCE: Enhanced evaluation caching
        self.enhanced_eval_cache = {}  # More aggressive caching
        self.eval_cache_hits = 0
        self.eval_cache_misses = 0
        self.max_eval_cache_size = 25000  # Larger cache
        
        # Advanced search infrastructure
        self.transposition_table: Dict[int, TranspositionEntry] = {}
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        self.zobrist = ZobristHashing()
        
        # V11 PHASE 2: Nudge System Integration
        self.nudge_database = {}
        self.nudge_stats = {
            'hits': 0,
            'misses': 0,
            'moves_boosted': 0,
            'positions_matched': 0,
            'instant_moves': 0,      # V11 PHASE 2 ENHANCEMENT
            'instant_time_saved': 0.0  # V11 PHASE 2 ENHANCEMENT
        }
        
        # V11 PHASE 2 ENHANCEMENT: Nudge threshold configuration
        self.nudge_instant_config = {
            'min_frequency': 8,        # Move must be played at least 8 times
            'min_eval': 0.4,          # Move must have eval improvement >= 0.4
            'confidence_threshold': 12.0  # Combined confidence score threshold
        }
        
        self._load_nudge_database()
        
        # Configuration
        self.max_tt_entries = 50000  # Reasonable size for testing
        
        # V11.3 PERFORMANCE: Piece position caching to reduce piece_at() calls
        self.piece_position_cache = {}
        self.board_hash_cache = None
        
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
               alpha: float = -99999, beta: float = 99999, is_root: bool = True) -> Tuple[chess.Move, float, dict]:
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
        
        V11.5 PERFORMANCE FIX: Now returns (move, score, search_info) tuple for compatibility
        """
        
        # ROOT LEVEL: Iterative deepening with time management
        if is_root:
            self.nodes_searched = 0
            self.search_start_time = time.time()
            
            # Helper function for early returns
            def _create_search_info(move, score=0, depth=0, nodes=0):
                total_time = time.time() - self.search_start_time
                return move, score, {
                    'nodes': nodes,
                    'time': total_time,
                    'nps': nodes / max(total_time, 0.001),
                    'depth': depth,
                    'score': score
                }
            
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return _create_search_info(chess.Move.null())

            # PV FOLLOWING OPTIMIZATION - check if current position triggers instant move
            pv_move = self.pv_tracker.check_position_for_instant_move(board)
            if pv_move:
                return _create_search_info(pv_move, 50, 1, 1)
            
            # V11 PHASE 2 ENHANCEMENT: Check for instant nudge moves (high confidence)
            instant_nudge_move = self._check_instant_nudge_move(board)
            if instant_nudge_move:
                # Calculate time saved
                time_saved = time_limit * 0.8  # Estimate time that would have been used
                self.nudge_stats['instant_time_saved'] += time_saved
                
                # Output instant move info
                print(f"info depth NUDGE score cp 50 nodes 0 time 0 pv {instant_nudge_move}")
                print(f"info string Instant nudge move: {instant_nudge_move} (high confidence)")
                
                return _create_search_info(instant_nudge_move, 50, 1, 1)
            
            # V11 ENHANCEMENT: Adaptive time management
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
            
            # V11.5 PERFORMANCE FIX: Return proper tuple format for compatibility
            total_time = time.time() - self.search_start_time
            search_info = {
                'nodes': self.nodes_searched,
                'time': total_time,
                'nps': self.nodes_searched / max(total_time, 0.001),
                'depth': current_depth,
                'score': best_score
            }
            return best_move, best_score, search_info
        
        # This should never be called directly with is_root=False from external code
        else:
            # Fallback - call the recursive search method
            score, move = self._recursive_search(board, depth or 1, alpha, beta, time_limit)
            final_move = move if move else chess.Move.null()
            total_time = time.time() - self.search_start_time if hasattr(self, 'search_start_time') else 0.001
            search_info = {
                'nodes': self.nodes_searched,
                'time': total_time,
                'nps': self.nodes_searched / max(total_time, 0.001),
                'depth': depth or 1,
                'score': score
            }
            return final_move, score, search_info
    
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
                return self._evaluate_position(board, depth=self.default_depth - search_depth), None
        
        # 1. TRANSPOSITION TABLE PROBE
        tt_hit, tt_score, tt_move = self._probe_transposition_table(board, search_depth, int(alpha), int(beta))
        if tt_hit:
            return float(tt_score), tt_move
        
        # 2. TERMINAL CONDITIONS
        if search_depth == 0:
            # V11.4: Use depth-aware evaluation instead of separate quiescence search
            score = self._evaluate_position(board, depth=self.default_depth - search_depth + 4)
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
            # Calculate nudge bonus for this move (V11 PHASE 2)
            nudge_bonus = self._get_nudge_bonus(board, move)
            
            # 1. Transposition table move (highest priority)
            if tt_move and move == tt_move:
                tt_moves.append(move)
            
            # 2. Nudge moves (second highest priority - V11 PHASE 2)
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
                
                # V11.4: Depth-aware tactical analysis for move ordering
                if depth <= 4:  # Only do full tactical analysis at shallow depths
                    tactical_bonus = self._detect_bitboard_tactics(board, move)
                else:
                    tactical_bonus = 0  # Skip tactical analysis at deep depths
                total_score = mvv_lva_score + tactical_bonus
                
                captures.append((total_score, move))
            
            # 4. Checks (high priority for tactical play)
            elif board.gives_check(move):
                # V11.4: Depth-aware tactical analysis for checking moves
                if depth <= 4:
                    tactical_bonus = self._detect_bitboard_tactics(board, move)
                else:
                    tactical_bonus = 20.0  # Small fixed bonus for checks at deep depths
                checks.append((tactical_bonus, move))
            
            # 5. Killer moves
            elif move in killer_set:
                killers.append(move)
                self.search_stats['killer_hits'] += 1
            
            # 6. Check for tactical patterns in quiet moves
            else:
                history_score = self.history_heuristic.get_history_score(move)
                
                # V11.4: Skip expensive tactical analysis for quiet moves at deep depths
                if depth <= 4:
                    tactical_bonus = self._detect_bitboard_tactics(board, move)
                else:
                    tactical_bonus = 0  # No tactical analysis for quiet moves at deep depths
                
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
    
    def _detect_position_tone(self, board: chess.Board) -> str:
        """V11.4: Detect position tone for dynamic evaluation selection"""
        try:
            # Basic material analysis
            white_material = sum(self.piece_values[piece.piece_type] for piece in board.piece_map().values() if piece.color == chess.WHITE)
            black_material = sum(self.piece_values[piece.piece_type] for piece in board.piece_map().values() if piece.color == chess.BLACK)
            
            current_player_material = white_material if board.turn else black_material
            opponent_material = black_material if board.turn else white_material
            material_balance = current_player_material - opponent_material
            
            # Check for immediate threats (king in check, pieces under attack)
            is_in_check = board.is_check()
            king_square = board.king(board.turn)
            attackers_count = len(board.attackers(not board.turn, king_square)) if king_square else 0
            
            # Determine position tone
            if is_in_check or material_balance < -200 or attackers_count > 2:
                return "defensive"
            elif material_balance > 200 and not is_in_check:
                return "offensive"
            else:
                return "neutral"
                
        except Exception:
            return "neutral"  # Fallback to neutral if analysis fails
    
    def _get_evaluation_components(self, depth: int, position_tone: str, time_pressure: bool = False) -> List[str]:
        """V11.4: Determine which evaluation components to include based on depth and position"""
        if time_pressure:
            return ["critical"]
        
        components = ["critical"]  # Always include critical components
        
        # Depth-based component inclusion
        if depth <= 6:
            components.append("primary")
        if depth <= 4:
            components.append("secondary")
        if depth <= 2:
            components.append("tertiary")
            
        # Position tone adjustments
        if position_tone == "defensive" and depth <= 8:
            # Continue deeper evaluation for safety
            if "secondary" not in components:
                components.append("secondary")
        elif position_tone == "offensive":
            # Lean forward, reduce complexity
            if "tertiary" in components:
                components.remove("tertiary")
                
        return components

    def _evaluate_position(self, board: chess.Board, depth: int = 0) -> float:
        """V11.4 DIMINISHING EVALUATIONS: Depth-aware position evaluation"""
        # V11.3 CRITICAL: Enhanced evaluation caching
        position_hash = hash(board.fen())
        if position_hash in self.enhanced_eval_cache:
            self.eval_cache_hits += 1
            return self.enhanced_eval_cache[position_hash]
        
        self.eval_cache_misses += 1
        
        # Create cache key
        cache_key = board.fen()
        
        if cache_key in self.evaluation_cache:
            self.search_stats['cache_hits'] += 1
            return self.evaluation_cache[cache_key]
        
        self.search_stats['cache_misses'] += 1
        
        # V11.4: Detect position tone and determine evaluation components
        position_tone = self._detect_position_tone(board)
        time_pressure = hasattr(self, 'time_remaining') and getattr(self, 'time_remaining', 10.0) < 2.0
        components = self._get_evaluation_components(depth, position_tone, time_pressure)
        
        # CRITICAL EVALUATIONS (Always computed)
        white_base = self.bitboard_evaluator.calculate_score_optimized(board, True)
        black_base = self.bitboard_evaluator.calculate_score_optimized(board, False)
        
        # Initialize totals with critical evaluation
        white_total = white_base
        black_total = black_base
        
        try:
            # PRIMARY EVALUATIONS (Depth <= 6)
            if "primary" in components:
                white_pawn_score = self.advanced_pawn_evaluator.evaluate_pawn_structure(board, True)
                black_pawn_score = self.advanced_pawn_evaluator.evaluate_pawn_structure(board, False)
                
                white_king_score = self.king_safety_evaluator.evaluate_king_safety(board, True)
                black_king_score = self.king_safety_evaluator.evaluate_king_safety(board, False)
                
                # Apply phase-aware weighting to primary components
                white_adjusted, black_adjusted = self._apply_phase_aware_weighting(
                    board, white_base, black_base, white_pawn_score, black_pawn_score,
                    white_king_score, black_king_score
                )
                white_total = white_adjusted
                black_total = black_adjusted
            
            # SECONDARY EVALUATIONS (Depth <= 4)
            if "secondary" in components:
                # Tactical pattern evaluation - but only at reasonable depths
                if depth <= 4 or (depth <= 6 and position_tone == "defensive"):
                    white_tactical_score = self._get_cached_tactical_evaluation(board, True)
                    black_tactical_score = self._get_cached_tactical_evaluation(board, False)
                    white_total += white_tactical_score
                    black_total += black_tactical_score
                
                # Enhanced endgame king evaluation
                endgame_king_bonus = self._evaluate_enhanced_endgame_king(board)
                move_classification_bonus = self._evaluate_move_classification_bonuses(board)
                
                if board.turn:  # White to move
                    white_total += endgame_king_bonus + move_classification_bonus
                else:  # Black to move
                    black_total += endgame_king_bonus + move_classification_bonus
            
            # TERTIARY EVALUATIONS (Depth <= 2)
            if "tertiary" in components:
                # Deep analysis components
                draw_penalty = self._evaluate_draw_penalty(board)
                draw_position_bonus = self._evaluate_draw_position_heuristic(board)
                king_restriction_bonus = self._evaluate_king_restriction_bonus(board)
                
                # Apply phase-aware multipliers to tertiary components
                phase_multipliers = self._get_phase_bonus_multipliers(board)
                weighted_draw_penalty = draw_penalty * phase_multipliers['draw_penalty']
                weighted_king_restriction = king_restriction_bonus * phase_multipliers['king_restriction']
                
                # Add tertiary scores
                white_total += draw_position_bonus
                black_total += draw_position_bonus
                
                if board.turn:  # White to move
                    white_total += weighted_draw_penalty + weighted_king_restriction
                else:  # Black to move
                    black_total += weighted_draw_penalty + weighted_king_restriction
                    
        except Exception as e:
            # Fallback to critical evaluation only if advanced evaluation fails
            white_total = white_base
            black_total = black_base
            
            # Still apply basic heuristics in fallback if not at deep depth
            if depth <= 4:
                try:
                    endgame_king_bonus = self._evaluate_enhanced_endgame_king(board)
                    if board.turn:  # White to move
                        white_total += endgame_king_bonus
                    else:  # Black to move
                        black_total += endgame_king_bonus
                except:
                    pass  # Ignore heuristic errors in fallback
        
        # Calculate final score from current player's perspective
        if board.turn:  # White to move
            final_score = white_total - black_total
        else:  # Black to move
            final_score = black_total - white_total
        
        # Cache the result with depth awareness
        self.evaluation_cache[cache_key] = final_score
        self.enhanced_eval_cache[position_hash] = final_score
        return final_score
    
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
    
    def _get_cached_tactical_evaluation(self, board: chess.Board, color: chess.Color) -> float:
        """
        V11.5 PERFORMANCE FIX: Cached tactical evaluation to eliminate redundant pattern detection
        """
        # Check cache first
        cached_result = self.tactical_cache.get_cached_result(board, color)
        if cached_result is not None:
            return cached_result
        
        # Original tactical detection code
        tactical_score = self.tactical_pattern_detector.evaluate_tactical_patterns(board, color)
        
        # Cache the result
        self.tactical_cache.cache_result(board, tactical_score, color)
        
        return tactical_score

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
            
            # V11.5 PERFORMANCE FIX: Use cached tactical evaluation
            tactical_score = self._get_cached_tactical_evaluation(board, our_color)
            tactical_bonus += tactical_score * 0.1  # Scale down for move ordering
            
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

    # V11 PHASE 2: NUDGE SYSTEM METHODS
    
    def _load_nudge_database(self):
        """Load the enhanced nudge database from JSON file"""
        try:
            # Construct path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Try enhanced database first, fallback to original
            enhanced_db_path = os.path.join(current_dir, 'v7p3r_enhanced_nudges.json')
            original_db_path = os.path.join(current_dir, 'v7p3r_nudge_database.json')
            
            if os.path.exists(enhanced_db_path):
                with open(enhanced_db_path, 'r', encoding='utf-8') as f:
                    self.nudge_database = json.load(f)
                
                # Count tactical positions
                tactical_count = sum(1 for pos in self.nudge_database.values() 
                                   for move in pos['moves'].values() 
                                   if move.get('source') == 'puzzle')
                
                print(f"info string Loaded enhanced nudge database: {len(self.nudge_database)} positions ({tactical_count} tactical)")
                
            elif os.path.exists(original_db_path):
                with open(original_db_path, 'r', encoding='utf-8') as f:
                    self.nudge_database = json.load(f)
                print(f"info string Loaded original nudge database: {len(self.nudge_database)} positions")
            else:
                print(f"info string No nudge database found")
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
        """Calculate enhanced nudge bonus for a move in current position"""
        try:
            position_key = self._get_position_key(board)
            
            # Check if position exists in nudge database
            if position_key not in self.nudge_database:
                self.nudge_stats['misses'] += 1
                return 0.0
            
            position_data = self.nudge_database[position_key]
            move_uci = move.uci()
            
            # Check if move exists in nudge data
            if 'moves' not in position_data or move_uci not in position_data['moves']:
                return 0.0
            
            move_data = position_data['moves'][move_uci]
            
            # Enhanced bonus calculation using new schema
            base_bonus = 50.0
            
            # Use confidence score if available (enhanced format)
            if 'confidence' in move_data:
                confidence = move_data['confidence']
                base_bonus *= (1.0 + confidence)  # Scale by confidence
            
            # Use frequency and evaluation
            frequency = move_data.get('frequency', 1)
            evaluation = move_data.get('eval', 0.0)
            
            # Frequency multiplier (more frequent = higher bonus, capped at 3x)
            frequency_multiplier = min(frequency / 2.0, 3.0)
            
            # Evaluation multiplier (better evaluation = higher bonus)
            eval_multiplier = max(evaluation / 0.5, 1.0) if evaluation > 0 else 1.0
            
            # Enhanced tactical bonuses
            tactical_info = move_data.get('tactical_info', {})
            tactical_bonus = 1.0
            
            if tactical_info:
                classification = tactical_info.get('classification', 'development')
                
                # Bonus based on tactical classification
                if classification == 'offensive':
                    tactical_bonus = 1.5  # Higher bonus for tactical strikes
                elif classification == 'defensive':
                    tactical_bonus = 1.2  # Moderate bonus for defensive moves
                
                # Additional bonus for puzzle-derived moves
                if move_data.get('source') == 'puzzle' or move_data.get('source') == 'hybrid':
                    tactical_bonus *= 1.3  # Proven tactical knowledge bonus
            
            total_bonus = base_bonus * frequency_multiplier * eval_multiplier * tactical_bonus
            
            # Update statistics
            self.nudge_stats['hits'] += 1
            self.nudge_stats['moves_boosted'] += 1
            
            return total_bonus
            
        except Exception as e:
            # Silently handle errors to avoid disrupting search
            return 0.0

    def _check_instant_nudge_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        V11 PHASE 2 ENHANCEMENT: Check for instant nudge moves that bypass search
        Enhanced with confidence scores and tactical awareness
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
                    
                    # Enhanced confidence calculation
                    if 'confidence' in move_data:
                        # Use actual confidence score from enhanced database
                        move_confidence = move_data['confidence']
                        
                        # Boost confidence for tactical moves
                        tactical_info = move_data.get('tactical_info', {})
                        if tactical_info.get('classification') == 'offensive':
                            move_confidence *= 1.2
                        
                        # Extra boost for puzzle-derived moves
                        if move_data.get('source') in ['puzzle', 'hybrid']:
                            move_confidence *= 1.3
                    else:
                        # Fallback to old calculation
                        frequency = move_data.get('frequency', 0)
                        evaluation = move_data.get('eval', 0.0)
                        
                        # Check minimum thresholds
                        if (frequency < self.nudge_instant_config['min_frequency'] or 
                            evaluation < self.nudge_instant_config['min_eval']):
                            continue
                        
                        move_confidence = (frequency + evaluation * 10) / 20.0  # Normalize to 0-1
                    
                    # Track best candidate
                    if move_confidence > best_confidence:
                        best_confidence = move_confidence
                        best_move = move
                
                except:
                    continue
            
            # Check if best move meets confidence threshold (adjusted for 0-1 scale)
            confidence_threshold = 0.9  # High threshold for instant moves
            
            if best_move and best_confidence >= confidence_threshold:
                # Update statistics
                self.nudge_stats['instant_moves'] += 1
                
                return best_move
            
            return None
            
        except Exception as e:
            # Silently handle errors to avoid disrupting search
            return None

    # V11.3 PHASE 1: DRAW PENALTY HEURISTICS
    def _evaluate_draw_penalty(self, board: chess.Board) -> float:
        """
        V11.3: Evaluate draw tendency penalty to encourage decisive play.
        Returns penalty value (negative score for draw-like positions)
        """
        penalty = 0.0
        
        # 1. REPETITION PENALTY: Discourage immediate repetition
        if len(board.move_stack) >= 4:
            # Check for position repetition in recent moves
            try:
                current_fen = board.fen().split(' ')[0]  # Position only
                
                # Create a copy to avoid modifying the original board
                temp_board = board.copy()
                
                # Check last 2 moves for immediate repetition
                temp_board.pop()
                temp_board.pop()
                prev_fen = temp_board.fen().split(' ')[0]
                
                if current_fen == prev_fen:
                    penalty -= 15.0  # Mild penalty for repetition
                    
            except Exception:
                pass  # Ignore errors, keep searching
        
        # 2. STALEMATE AVOIDANCE: Detect potential stalemate setups
        if board.is_stalemate():
            penalty -= 50.0  # Strong penalty for actual stalemate
        elif len(list(board.legal_moves)) <= 3:
            # Few legal moves - potential stalemate risk
            penalty -= 10.0
            
        # 3. INSUFFICIENT MATERIAL PENALTY: Encourage trades when winning
        white_material = self._count_material(board, chess.WHITE)
        black_material = self._count_material(board, chess.BLACK)
        material_diff = abs(white_material - black_material)
        
        # If one side has clear material advantage but insufficient to mate
        if material_diff > 200:  # Clear advantage
            total_material = white_material + black_material
            if total_material < 1000:  # Low material endgame
                # Encourage keeping material for mating attack
                penalty -= 5.0
        
        # 4. FIFTY MOVE RULE AWARENESS: Slight penalty as we approach 50-move rule
        halfmove_clock = board.halfmove_clock
        if halfmove_clock > 30:  # Getting close to 50-move rule
            # Gradually increase penalty
            penalty -= (halfmove_clock - 30) * 0.5
            
        return penalty
    
    def _evaluate_draw_position_heuristic(self, board: chess.Board) -> float:
        """
        V11.3: Additional draw-aware position evaluation
        """
        bonus = 0.0
        
        # Encourage piece activity in simplified positions
        piece_count = len(board.piece_map())
        if piece_count <= 12:  # Simplified endgame
            # Count active pieces (not on starting squares or edge)
            active_pieces = 0
            for square, piece in board.piece_map().items():
                if piece.piece_type != chess.PAWN:
                    # Consider piece active if not on back rank
                    rank = chess.square_rank(square)
                    if piece.color == chess.WHITE and rank > 1:
                        active_pieces += 1
                    elif piece.color == chess.BLACK and rank < 6:
                        active_pieces += 1
            
            # Bonus for active piece play
            bonus += active_pieces * 3.0
            
        return bonus

    # V11.3 PHASE 2: ENHANCED ENDGAME KING EVALUATION
    def _evaluate_enhanced_endgame_king(self, board: chess.Board) -> float:
        """
        V11.3: Enhanced endgame king evaluation focusing on activity, centralization,
        and pawn support when material is low.
        """
        # Check if we're in an endgame
        total_material = self._count_material(board, chess.WHITE) + self._count_material(board, chess.BLACK)
        if total_material > 1500:  # Not endgame yet
            return 0.0
            
        bonus = 0.0
        
        # Evaluate both kings
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        if white_king_square is None or black_king_square is None:
            return 0.0
            
        # 1. KING CENTRALIZATION (more important in endgames)
        white_centralization = self._calculate_king_centralization_bonus(white_king_square)
        black_centralization = self._calculate_king_centralization_bonus(black_king_square)
        
        # 2. KING ACTIVITY (mobility in endgame)
        white_activity = self._calculate_king_activity_bonus(board, white_king_square, chess.WHITE)
        black_activity = self._calculate_king_activity_bonus(board, black_king_square, chess.BLACK)
        
        # 3. KING-PAWN COORDINATION
        white_pawn_support = self._calculate_king_pawn_support(board, white_king_square, chess.WHITE)
        black_pawn_support = self._calculate_king_pawn_support(board, black_king_square, chess.BLACK)
        
        # 4. OPPOSITION (basic king vs king positioning)
        opposition_bonus = self._calculate_opposition_bonus(board, white_king_square, black_king_square)
        
        # Combine bonuses from current player's perspective
        if board.turn:  # White to move
            bonus = (white_centralization - black_centralization) + \
                   (white_activity - black_activity) + \
                   (white_pawn_support - black_pawn_support) + \
                   opposition_bonus
        else:  # Black to move
            bonus = (black_centralization - white_centralization) + \
                   (black_activity - white_activity) + \
                   (black_pawn_support - white_pawn_support) - \
                   opposition_bonus
        
        return bonus
        
    def _calculate_king_centralization_bonus(self, king_square: int) -> float:
        """Calculate centralization bonus for a king in endgame"""
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Distance from center (files 3,4 and ranks 3,4 are most central)
        file_distance = min(abs(king_file - 3), abs(king_file - 4))
        rank_distance = min(abs(king_rank - 3), abs(king_rank - 4))
        
        # Higher bonus for more central positions
        centralization_score = 5 - (file_distance + rank_distance)
        return max(centralization_score * 4.0, 0.0)  # Scale for significant impact
        
    def _calculate_king_activity_bonus(self, board: chess.Board, king_square: int, color: chess.Color) -> float:
        """Calculate king activity (mobility) bonus"""
        # Save original turn
        original_turn = board.turn
        board.turn = color
        
        # Count legal king moves
        king_moves = len([move for move in board.legal_moves if move.from_square == king_square])
        
        # Restore turn
        board.turn = original_turn
        
        # Bonus for mobile king (up to 8 moves possible)
        return king_moves * 2.5
        
    def _calculate_king_pawn_support(self, board: chess.Board, king_square: int, color: chess.Color) -> float:
        """Calculate king-pawn coordination bonus"""
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        support_bonus = 0.0
        pawns = board.pieces(chess.PAWN, color)
        
        for pawn_square in pawns:
            pawn_file = chess.square_file(pawn_square)
            pawn_rank = chess.square_rank(pawn_square)
            
            # Distance between king and pawn
            distance = max(abs(king_file - pawn_file), abs(king_rank - pawn_rank))
            
            # Bonus for king near own pawns (support/escort)
            if distance <= 2:
                support_bonus += 3.0
            elif distance <= 3:
                support_bonus += 1.5
                
            # Special bonus for king in front of advanced pawns
            if color == chess.WHITE:
                if king_rank > pawn_rank and abs(king_file - pawn_file) <= 1 and pawn_rank >= 4:
                    support_bonus += 5.0  # King supporting pawn advance
            else:
                if king_rank < pawn_rank and abs(king_file - pawn_file) <= 1 and pawn_rank <= 3:
                    support_bonus += 5.0
                    
        return support_bonus
        
    def _calculate_opposition_bonus(self, board: chess.Board, white_king: int, black_king: int) -> float:
        """Calculate basic opposition bonus (white's perspective)"""
        white_file = chess.square_file(white_king)
        white_rank = chess.square_rank(white_king)
        black_file = chess.square_file(black_king)
        black_rank = chess.square_rank(black_king)
        
        file_diff = abs(white_file - black_file)
        rank_diff = abs(white_rank - black_rank)
        
        # Direct opposition (same file/rank, 2 squares apart)
        if (file_diff == 0 and rank_diff == 2) or (rank_diff == 0 and file_diff == 2):
            # Bonus for player to move (they have the opposition)
            return 8.0 if board.turn else -8.0
            
        # Distant opposition (same file/rank, even number of squares apart)
        if file_diff == 0 and rank_diff > 2 and rank_diff % 2 == 0:
            return 4.0 if board.turn else -4.0
        if rank_diff == 0 and file_diff > 2 and file_diff % 2 == 0:
            return 4.0 if board.turn else -4.0
            
        return 0.0

    # V11.3 PHASE 3: MOVE CLASSIFICATION SYSTEM
    def _evaluate_move_classification_bonuses(self, board: chess.Board) -> float:
        """
        V11.3: Evaluate move classification bonuses to improve strategic decision-making.
        Classifies moves as offensive, defensive, or developing and applies phase-appropriate bonuses.
        """
        if len(board.move_stack) == 0:
            return 0.0  # No moves to analyze
            
        last_move = board.move_stack[-1]
        bonus = 0.0
        
        # Determine game phase
        total_material = self._count_material(board, chess.WHITE) + self._count_material(board, chess.BLACK)
        
        if total_material > 2500:  # Opening phase
            bonus += self._evaluate_developing_moves(board, last_move)
        elif total_material > 1500:  # Middlegame phase  
            bonus += self._evaluate_offensive_moves(board, last_move)
            bonus += self._evaluate_defensive_moves(board, last_move)
        else:  # Endgame phase
            bonus += self._evaluate_endgame_moves(board, last_move)
            
        return bonus
        
    def _evaluate_developing_moves(self, board: chess.Board, move: chess.Move) -> float:
        """Bonus for good developing moves in opening"""
        bonus = 0.0
        piece = board.piece_at(move.to_square)
        
        if piece is None:
            return 0.0
            
        # Knight and bishop development
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            from_rank = chess.square_rank(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            
            # Bonus for moving pieces off back rank
            if piece.color == chess.WHITE:
                if from_rank == 0 and to_rank > 0:
                    bonus += 8.0  # Good development
                # Extra bonus for central development
                to_file = chess.square_file(move.to_square)
                if to_file in [2, 3, 4, 5]:  # Central files
                    bonus += 4.0
            else:
                if from_rank == 7 and to_rank < 7:
                    bonus += 8.0
                to_file = chess.square_file(move.to_square)
                if to_file in [2, 3, 4, 5]:
                    bonus += 4.0
                    
        # Castling bonus
        if board.is_castling(move):
            bonus += 15.0
            
        # Central pawn moves
        if piece.piece_type == chess.PAWN:
            to_file = chess.square_file(move.to_square)
            if to_file in [3, 4]:  # d and e files
                bonus += 5.0
                
        return bonus
        
    def _evaluate_offensive_moves(self, board: chess.Board, move: chess.Move) -> float:
        """Bonus for good offensive moves in middlegame"""
        bonus = 0.0
        
        # Captures
        if board.is_capture(move):
            bonus += 6.0
            
        # Checks
        if board.gives_check(move):
            bonus += 4.0
            
        # Attacks on enemy pieces
        piece_at_destination = board.piece_at(move.to_square)
        if piece_at_destination:
            # Bonus for attacking higher value pieces
            attacked_squares = list(board.attacks(move.to_square))
            for square in attacked_squares:
                enemy_piece = board.piece_at(square)
                if enemy_piece and enemy_piece.color != piece_at_destination.color:
                    if enemy_piece.piece_type == chess.QUEEN:
                        bonus += 3.0
                    elif enemy_piece.piece_type in [chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                        bonus += 2.0
                        
        return bonus
        
    def _evaluate_defensive_moves(self, board: chess.Board, move: chess.Move) -> float:
        """Bonus for good defensive moves"""
        bonus = 0.0
        piece = board.piece_at(move.to_square)
        
        if piece is None:
            return 0.0
            
        # Defending own pieces
        defended_squares = list(board.attacks(move.to_square))
        for square in defended_squares:
            friendly_piece = board.piece_at(square)
            if friendly_piece and friendly_piece.color == piece.color:
                # Bonus for defending valuable pieces
                if friendly_piece.piece_type == chess.QUEEN:
                    bonus += 4.0
                elif friendly_piece.piece_type in [chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    bonus += 2.0
                elif friendly_piece.piece_type == chess.KING:
                    bonus += 3.0
                    
        # King safety moves
        if piece.piece_type == chess.KING:
            king_square = move.to_square
            # Bonus for moving king to safety
            enemy_attackers = len(list(board.attackers(not piece.color, king_square)))
            if enemy_attackers == 0:
                bonus += 5.0  # Safe king move
                
        return bonus
        
    def _evaluate_endgame_moves(self, board: chess.Board, move: chess.Move) -> float:
        """Bonus for good endgame moves"""
        bonus = 0.0
        piece = board.piece_at(move.to_square)
        
        if piece is None:
            return 0.0
            
        # King activity
        if piece.piece_type == chess.KING:
            bonus += 4.0  # Active king is good in endgame
            
        # Pawn promotion threats
        if piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(move.to_square)
            if piece.color == chess.WHITE:
                if to_rank >= 6:  # 7th or 8th rank
                    bonus += 8.0
            else:
                if to_rank <= 1:  # 2nd or 1st rank
                    bonus += 8.0
                    
        # Supporting passed pawns
        if piece.piece_type in [chess.KING, chess.ROOK]:
            # Check if move supports own pawns
            own_pawns = board.pieces(chess.PAWN, piece.color)
            for pawn_square in own_pawns:
                distance = max(abs(chess.square_file(move.to_square) - chess.square_file(pawn_square)),
                              abs(chess.square_rank(move.to_square) - chess.square_rank(pawn_square)))
                if distance <= 2:
                    bonus += 2.0  # Supporting pawn
                    
        return bonus

    # V11.3 PHASE 4: KING RESTRICTION "CLOSING THE BOX" HEURISTIC
    def _evaluate_king_restriction_bonus(self, board: chess.Board) -> float:
        """
        V11.3: "Closing the box" heuristic - bonus for moves that reduce opponent 
        king mobility in winning endgames, especially when ahead in material.
        """
        # Only apply in simplified positions where king mobility matters
        total_material = self._count_material(board, chess.WHITE) + self._count_material(board, chess.BLACK)
        if total_material > 1200:  # Not simplified enough
            return 0.0
            
        # Check if we have a material advantage
        white_material = self._count_material(board, chess.WHITE)
        black_material = self._count_material(board, chess.BLACK)
        material_diff = white_material - black_material
        
        # Only apply if we have a significant advantage
        if abs(material_diff) < 200:  # Not enough advantage
            return 0.0
            
        bonus = 0.0
        
        # Determine which side has the advantage
        if material_diff > 0:  # White has advantage
            # Evaluate black king restriction
            black_king = board.king(chess.BLACK)
            if black_king is not None:
                restriction_bonus = self._calculate_king_restriction_score(board, black_king, chess.BLACK)
                bonus += restriction_bonus if board.turn else -restriction_bonus
        else:  # Black has advantage
            # Evaluate white king restriction
            white_king = board.king(chess.WHITE)
            if white_king is not None:
                restriction_bonus = self._calculate_king_restriction_score(board, white_king, chess.WHITE)
                bonus += -restriction_bonus if board.turn else restriction_bonus
                
        return bonus
        
    def _calculate_king_restriction_score(self, board: chess.Board, king_square: int, king_color: chess.Color) -> float:
        """Calculate how restricted the king is (higher = more restricted)"""
        score = 0.0
        
        # 1. MOBILITY RESTRICTION: Count available moves
        original_turn = board.turn
        board.turn = king_color
        
        king_moves = [move for move in board.legal_moves if move.from_square == king_square]
        mobility_score = len(king_moves)
        
        board.turn = original_turn
        
        # Bonus for restricting king mobility (fewer moves = higher bonus)
        max_moves = 8  # King can have at most 8 moves
        restriction_score = (max_moves - mobility_score) * 3.0
        score += restriction_score
        
        # 2. EDGE/CORNER RESTRICTION: Bonus for driving king to edges
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Edge bonus
        if king_file == 0 or king_file == 7:  # a-file or h-file
            score += 8.0
        elif king_file == 1 or king_file == 6:  # b-file or g-file
            score += 4.0
            
        if king_rank == 0 or king_rank == 7:  # 1st or 8th rank
            score += 8.0
        elif king_rank == 1 or king_rank == 6:  # 2nd or 7th rank
            score += 4.0
            
        # Corner bonus (cumulative with edge)
        if ((king_file == 0 or king_file == 7) and (king_rank == 0 or king_rank == 7)):
            score += 12.0  # Big bonus for cornering the king
            
        # 3. MATING NET: Bonus for controlling escape squares
        escape_squares_controlled = 0
        for file_offset in [-1, 0, 1]:
            for rank_offset in [-1, 0, 1]:
                if file_offset == 0 and rank_offset == 0:
                    continue
                    
                target_file = king_file + file_offset
                target_rank = king_rank + rank_offset
                
                if 0 <= target_file <= 7 and 0 <= target_rank <= 7:
                    target_square = chess.square(target_file, target_rank)
                    
                    # Check if enemy controls this square
                    if board.is_attacked_by(not king_color, target_square):
                        escape_squares_controlled += 1
                        
        # Bonus for controlling king's escape squares
        score += escape_squares_controlled * 2.0
        
        # 4. OPPOSITION AND KING COORDINATION: Bonus for good king positioning when winning
        our_king = board.king(not king_color)
        if our_king is not None:
            our_king_file = chess.square_file(our_king)
            our_king_rank = chess.square_rank(our_king)
            
            file_distance = abs(king_file - our_king_file)
            rank_distance = abs(king_rank - our_king_rank)
            
            # Bonus for maintaining optimal distance (close but not too close)
            if file_distance == 2 and rank_distance <= 1:  # Good opposition
                score += 6.0
            elif rank_distance == 2 and file_distance <= 1:
                score += 6.0
            elif file_distance + rank_distance == 3:  # Knight's move away
                score += 4.0
                
        return score

    # V11.3 PHASE 5: PHASE-AWARE EVALUATION PRIORITIES  
    def _apply_phase_aware_weighting(self, board: chess.Board, white_total: float, black_total: float,
                                   white_pawn_score: float, black_pawn_score: float,
                                   white_king_score: float, black_king_score: float) -> tuple:
        """
        V11.3: Apply phase-aware weighting to evaluation components.
        Opening: emphasize development, Middlegame: emphasize tactics, Endgame: emphasize king activity
        """
        total_material = self._count_material(board, chess.WHITE) + self._count_material(board, chess.BLACK)
        
        # Determine game phase weights
        if total_material > 2500:  # Opening phase
            development_weight = 1.5
            pawn_structure_weight = 0.8
            king_safety_weight = 1.2
            tactical_weight = 1.0
        elif total_material > 1500:  # Middlegame phase
            development_weight = 0.7
            pawn_structure_weight = 1.3
            king_safety_weight = 1.0
            tactical_weight = 1.4
        else:  # Endgame phase
            development_weight = 0.5
            pawn_structure_weight = 1.1
            king_safety_weight = 0.6  # Less important in endgame
            tactical_weight = 1.2
            
        # Apply weights to specific components
        # Note: We're applying relative adjustments to the component scores
        adjusted_white_pawn = white_pawn_score * pawn_structure_weight
        adjusted_black_pawn = black_pawn_score * pawn_structure_weight
        
        # For king scores, we adjust based on phase (safety vs activity)
        adjusted_white_king = white_king_score * king_safety_weight  
        adjusted_black_king = black_king_score * king_safety_weight
        
        # Recalculate totals with phase-aware weights
        # We keep the base material score unchanged and adjust only the positional components
        white_material_base = self._count_material(board, chess.WHITE)
        black_material_base = self._count_material(board, chess.BLACK)
        
        adjusted_white_total = white_material_base + adjusted_white_pawn + adjusted_white_king
        adjusted_black_total = black_material_base + adjusted_black_pawn + adjusted_black_king
        
        return adjusted_white_total, adjusted_black_total
        
    def _update_piece_cache(self, board: chess.Board):
        """V11.3 OPTIMIZATION: Cache piece positions to reduce piece_at() calls"""
        board_hash = hash(board.fen())
        if self.board_hash_cache != board_hash:
            self.piece_position_cache.clear()
            
            # Cache all piece positions
            for square in range(64):
                piece = board.piece_at(square)
                if piece:
                    self.piece_position_cache[square] = piece
                    
            self.board_hash_cache = board_hash
    
    def _get_cached_piece(self, square: int):
        """V11.3 OPTIMIZATION: Get piece from cache instead of board.piece_at()"""
        return self.piece_position_cache.get(square, None)
    def _get_phase_bonus_multipliers(self, board: chess.Board) -> dict:
        """
        V11.3: Get phase-specific bonus multipliers for V11.3 heuristics
        """
        total_material = self._count_material(board, chess.WHITE) + self._count_material(board, chess.BLACK)
        
        if total_material > 2500:  # Opening
            return {
                'draw_penalty': 0.5,  # Less important in opening
                'endgame_king': 0.3,  # Much less important in opening
                'move_classification': 1.5,  # Very important for development
                'king_restriction': 0.1  # Almost irrelevant in opening
            }
        elif total_material > 1500:  # Middlegame  
            return {
                'draw_penalty': 1.0,  # Normal importance
                'endgame_king': 0.7,  # Some importance
                'move_classification': 1.2,  # Important for tactics
                'king_restriction': 0.6  # Some importance
            }
        else:  # Endgame
            return {
                'draw_penalty': 1.3,  # More important to avoid draws
                'endgame_king': 1.5,  # Very important in endgame
                'move_classification': 0.8,  # Less important
                'king_restriction': 1.4  # Very important for winning
            }
    
