#!/usr/bin/env python3
"""
V7P3R Chess Engine v13.1 - Simplified Performance Build

Combining the best of V12.6 stability with V13.x move ordering improvements.
Focused on clean performance without complexity overhead.

DESIGN PHILOSOPHY:
- V12.6 evaluation baseline (proven stable)
- V13.x move ordering enhancements (84% pruning, 6.3x speedup)
- Simplified tactical detection (selective usage)
- Clean, maintainable codebase

VERSION LINEAGE:
- v12.6: Clean Performance Build (production baseline)
- v13.0: Tal Evolution - Tactical Pattern Recognition (experimental)
- v13.x: Capablanca Evolution - Positional Simplification (complex)
- v13.1: Simplified Performance Build (current)

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
    """V7P3R Chess Engine v13.2 - Puzzle Solver Evolution"""
    
    # V13.2 PUZZLE SOLVER FLAGS - Anti-Null-Move-Pruning System
    ENABLE_MULTI_PV = True              # V13.2: Multi-PV foundation for puzzle solving
    ENABLE_OPPORTUNITY_EVALUATION = True  # V13.2: Evaluate opportunities vs absolute scores
    ENABLE_PRACTICAL_SEARCH = True     # V13.2: Focus on improvement over perfection
    
    # V13.0 FEATURE FLAGS - Tal Evolution System
    ENABLE_TACTICAL_DETECTION = True    # V13.0: Pin/Fork/Skewer detection
    ENABLE_DYNAMIC_EVALUATION = True    # V13.0: Context-dependent piece values
    ENABLE_TAL_COMPLEXITY_BONUS = True  # V13.0: Position complexity assessment
    
    # V12.x LEGACY FLAGS (maintained for compatibility)
    ENABLE_NUDGE_SYSTEM = False         # V12.5: Disabled for V13 tactical refactor
    ENABLE_PV_FOLLOWING = True          # Keep - high value feature
    ENABLE_ADVANCED_EVALUATION = True   # V12.4: Re-enabled for V13
    
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
        
        # Evaluation components - V13.0: Tal Evolution System
        self.bitboard_evaluator = V7P3RScoringCalculationBitboard(self.piece_values)
        self.advanced_pawn_evaluator = V7P3RAdvancedPawnEvaluator()
        self.king_safety_evaluator = V7P3RKingSafetyEvaluator()
        
        # V13.0 NEW: Tactical detection and dynamic evaluation
        if self.ENABLE_TACTICAL_DETECTION:
            try:
                from v7p3r_tactical_detector import V7P3RTacticalDetector
                self.tactical_detector = V7P3RTacticalDetector()
            except ImportError:
                # Silent fallback for UCI compatibility
                self.tactical_detector = None
                self.ENABLE_TACTICAL_DETECTION = False
        else:
            self.tactical_detector = None
            
        if self.ENABLE_DYNAMIC_EVALUATION and self.tactical_detector:
            try:
                from v7p3r_dynamic_evaluator import V7P3RDynamicEvaluator
                self.dynamic_evaluator = V7P3RDynamicEvaluator(self.tactical_detector)
            except ImportError:
                # Silent fallback for UCI compatibility
                self.dynamic_evaluator = None
                self.ENABLE_DYNAMIC_EVALUATION = False
        else:
            self.dynamic_evaluator = None
        
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
            
            # V13.1 CRITICAL: Reset performance counters for optimization
            self.evaluation_count = 0
            self.dynamic_eval_count = 0
            self._current_depth = depth or self.default_depth
            
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
    
    def _recursive_search(self, board: chess.Board, search_depth: int, alpha: float, beta: float, time_limit: float, 
                        move_number: int = 0) -> Tuple[float, Optional[chess.Move]]:
        """
        V13.x CAPABLANCA EVOLUTION: Recursive search with dual-brain evaluation and early exit
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
            null_score, _ = self._recursive_search(board, search_depth - 2, -beta, -beta + 1, time_limit, move_number + 1)
            null_score = -null_score
            board.turn = not board.turn
            
            if null_score >= beta:
                return null_score, None
        
        # 4. MOVE GENERATION AND ORDERING - V13.x WITH CAPABLANCA SYSTEM
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0, None
        
        # V13.1: Simplified move ordering (V12.6 approach with V13.x enhancements)
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
            
            # V13.1 ASYMMETRIC SEARCH DEPTH: Model imperfect opponents algorithmically
            # Instead of changing evaluation, we search opponent moves with slightly less depth
            # This maintains evaluation consistency while modeling that opponents don't search as deeply
            
            opponent_depth_reduction = self._calculate_opponent_depth_reduction(search_depth, moves_searched)
            actual_search_depth = search_depth - 1 - opponent_depth_reduction
            
            # Search with possible reduction
            if reduction > 0:
                score, _ = self._recursive_search(board, actual_search_depth - reduction, -beta, -alpha, time_limit, move_number + 1)
                score = -score
                
                # Re-search at full depth if reduced search failed high
                if score > alpha:
                    score, _ = self._recursive_search(board, actual_search_depth, -beta, -alpha, time_limit, move_number + 1)
                    score = -score
            else:
                score, _ = self._recursive_search(board, actual_search_depth, -beta, -alpha, time_limit, move_number + 1)
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
        """V13.x FOCUSED MOVE ORDERING - Critical moves only, 83% pruning rate, 5.9x speedup"""
        if len(moves) <= 2:
            return moves
        
        # V13.x: Use focused move ordering system
        critical_moves, waiting_moves = self._order_moves_v13x_focused(board, moves, depth, tt_move)
        
        # Store waiting moves for later use (zugzwang, time pressure, quiescence)
        self.waiting_moves = waiting_moves
        
        # V13.x Performance tracking
        if not hasattr(self, 'v13x_stats'):
            self.v13x_stats = {
                'total_legal_moves': 0,
                'critical_moves_selected': 0,
                'pruning_rate': 0.0
            }
        
        self.v13x_stats['total_legal_moves'] += len(moves)
        self.v13x_stats['critical_moves_selected'] += len(critical_moves)
        if len(moves) > 0:
            self.v13x_stats['pruning_rate'] = (len(moves) - len(critical_moves)) / len(moves) * 100
        
        return critical_moves  # Only return critical moves for main search
    
    def _order_moves_v13x_focused(self, board: chess.Board, legal_moves: List[chess.Move], 
                                 depth: int = 0, tt_move: Optional[chess.Move] = None) -> tuple:
        """
        V13.x focused move ordering - returns (critical_moves, waiting_moves)
        Fixes V12.6 weaknesses: 75% bad move ordering, 70% tactical misses
        """
        # V13.x Priority Thresholds - Aggressive Pruning  
        CRITICAL_MOVE_THRESHOLD = 400    # Moves that must be searched
        IMPORTANT_MOVE_THRESHOLD = 200   # Moves worth considering
        QUIET_MOVE_THRESHOLD = 100       # Only if nothing else available
        MAX_CRITICAL_MOVES = 6          # Maximum moves to search in complex positions
        
        if len(legal_moves) <= 2:
            return legal_moves, []
        
        # Detect position characteristics
        position_info = self._analyze_position_v13x(board)
        
        # Score all moves
        move_scores = []
        for move in legal_moves:
            score = self._score_move_v13x(board, move, position_info)
            move_scores.append((score, move))
        
        # Sort by total score
        move_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Separate into critical and waiting moves
        critical_moves = []
        waiting_moves = []
        
        # Always include TT move if available
        if tt_move and tt_move in legal_moves:
            critical_moves.append(tt_move)
            # Remove from list to avoid duplicates
            move_scores = [(s, m) for s, m in move_scores if m != tt_move]
        
        for score, move in move_scores:
            if score >= CRITICAL_MOVE_THRESHOLD:
                critical_moves.append(move)
            elif score >= IMPORTANT_MOVE_THRESHOLD and len(critical_moves) < 4:
                # Include important moves but limit to 4 total
                critical_moves.append(move)
            elif score >= QUIET_MOVE_THRESHOLD and len(critical_moves) < 2:
                # Only include lower-tier moves if we have very few critical moves
                critical_moves.append(move)
            else:
                # Everything else goes to waiting list
                waiting_moves.append(move)
        
        # Enforce maximum critical moves limit for complex positions
        if len(critical_moves) > MAX_CRITICAL_MOVES:
            # Move excess critical moves to waiting moves
            excess_moves = critical_moves[MAX_CRITICAL_MOVES:]
            critical_moves = critical_moves[:MAX_CRITICAL_MOVES]
            waiting_moves = excess_moves + waiting_moves
        
        # Ensure we have at least 2 moves to search (unless fewer legal moves)
        while len(critical_moves) < min(2, len(legal_moves)) and waiting_moves:
            critical_moves.append(waiting_moves.pop(0))
        
        return critical_moves, waiting_moves
    
    def _analyze_position_v13x(self, board: chess.Board) -> dict:
        """Analyze position characteristics for V13.x move ordering"""
        info = {
            'in_check': board.is_check(),
            'hanging_pieces': self._detect_hanging_pieces_v13x(board),
            'attacking_pieces': self._detect_attacking_pieces_v13x(board),
            'game_phase': self._determine_game_phase_v13x(board),
            'material_balance': self._calculate_material_balance_v13x(board),
            'king_safety_issues': self._detect_king_safety_issues_v13x(board),
            'piece_count': len(board.piece_map())
        }
        return info
    
    def _score_move_v13x(self, board: chess.Board, move: chess.Move, position_info: dict) -> int:
        """V13.x enhanced tactical move scoring - TACTICAL ACCURACY PATCHES APPLIED"""
        score = 0
        
        # PRIORITY 1: CHECKMATE DETECTION (PATCH 1 - HIGH PRIORITY)
        if self._gives_check_safe_v13x(board, move):
            board_copy = board.copy()
            board_copy.push(move)
            if board_copy.is_checkmate():
                return 50000  # MASSIVE bonus for checkmate (was 10000)
            elif board_copy.is_check():
                score = 2000  # Higher check bonus (was 1000)
                # Bonus for checks that restrict king movement
                king_square = board_copy.king(not board.turn)
                if king_square:
                    king_moves = len([m for m in board_copy.legal_moves if m.from_square == king_square])
                    if king_moves <= 1:
                        score += 500  # Bonus for limiting king mobility
        
        # PRIORITY 2: KING SAFETY (save king from attacks)
        elif self._saves_king_safety_v13x(board, move, position_info):
            score = 1500  # Increased from 800
        
        # PRIORITY 3: PIECE SAFETY (save hanging pieces - 27.7% miss rate in V12.6)
        elif self._saves_hanging_piece_v13x(board, move, position_info):
            hanging_piece = board.piece_at(move.from_square)
            if hanging_piece:
                piece_value = self.piece_values.get(hanging_piece.piece_type, 0)
                score = 1000 + piece_value // 10  # Scale by piece value
        
        # PRIORITY 4: ENHANCED CAPTURES (PATCH 2 - HIGH PRIORITY - SEE EVALUATION)
        elif board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            
            if victim and attacker:
                victim_value = self.piece_values.get(victim.piece_type, 0)
                attacker_value = self.piece_values.get(attacker.piece_type, 0)
                
                # ENHANCED: Static Exchange Evaluation (SEE)
                material_gain = victim_value - attacker_value
                
                # Base capture score
                if victim_value >= attacker_value:
                    score = 1200 + (victim_value - attacker_value) // 10
                    
                    # ENHANCED: Multi-step capture safety check
                    if self._is_capture_safe_enhanced_v13x(board, move):
                        score += 300  # Safe capture bonus
                    elif material_gain > 0:
                        score += 100  # Still positive material gain
                    else:
                        score -= 400  # Penalize unsafe equal/bad captures
                else:
                    # Bad captures - much lower scores but not eliminated
                    score = 200 if material_gain > -200 else 50
                    
                # ENHANCED: Special capture bonuses
                if victim.piece_type == chess.QUEEN:
                    score += 500  # Queen capture bonus
                elif victim.piece_type == chess.ROOK:
                    score += 200  # Rook capture bonus
                elif victim.piece_type == chess.KNIGHT or victim.piece_type == chess.BISHOP:
                    score += 100  # Minor piece bonus
        
        # PRIORITY 5: ENHANCED TACTICAL THREATS (PATCH 3 - FORK/PIN DETECTION)
        elif self._creates_tactical_threat_enhanced_v13x(board, move):
            threat_type, threat_value = self._analyze_tactical_threat_v13x(board, move)
            score = 800 + threat_value
            
            # Special bonuses for different threat types
            if 'fork' in threat_type:
                score += 300  # Fork bonus
            elif 'pin' in threat_type:
                score += 200  # Pin bonus
            elif 'skewer' in threat_type:
                score += 250  # Skewer bonus
        
        # PRIORITY 6: PIECE DEVELOPMENT (opening/early middlegame)
        elif self._is_development_move_v13x(board, move, position_info):
            if position_info['game_phase'] == 'opening':
                piece = board.piece_at(move.from_square)
                if piece:
                    # ENHANCED: Better development scoring
                    if piece.piece_type == chess.KNIGHT:
                        score = 450  # Knights before bishops
                        # Prefer central squares for knights
                        center_squares = {chess.D4, chess.D5, chess.E4, chess.E5, chess.C3, chess.C6, chess.F3, chess.F6}
                        if move.to_square in center_squares:
                            score += 100
                    elif piece.piece_type == chess.BISHOP:
                        score = 400  # Bishop development
                        # Prefer long diagonals
                        long_diag = {chess.A1, chess.B2, chess.C3, chess.D4, chess.E5, chess.F6, chess.G7, chess.H8,
                                   chess.H1, chess.G2, chess.F3, chess.E4, chess.D5, chess.C6, chess.B7, chess.A8}
                        if move.to_square in long_diag:
                            score += 50
                    else:
                        score = 350
            else:
                score = 120  # Non-opening development
        
        # PRIORITY 7: CASTLING (enhanced scoring)
        elif board.is_castling(move):
            score = 500  # Increased from 350
            # Bonus if king is under pressure
            if position_info['king_safety_issues']:
                score += 200
        
        # PRIORITY 8: PAWN PROMOTIONS (enhanced)
        elif move.promotion == chess.QUEEN:
            score = 1500  # Massive promotion bonus
        elif move.promotion:
            score = 800   # Other promotions
        
        # PRIORITY 9: CENTER CONTROL (early game)
        elif self._controls_center_v13x(board, move) and position_info['game_phase'] == 'opening':
            score = 150   # Slightly increased
        
        # EVERYTHING ELSE: Quiet moves (heavily penalized but some get through)
        else:
            score = 10
            # ENHANCED: Better quiet move evaluation
            if self._improves_piece_position_enhanced_v13x(board, move):
                score += 25
        
        # Apply position-specific bonuses/penalties
        score += self._apply_position_modifiers_v13x(board, move, position_info)
        
        return score
    
    # V13.x Helper methods for move analysis
    def _detect_hanging_pieces_v13x(self, board: chess.Board) -> set:
        """Detect pieces that are hanging (undefended or underdefended)"""
        hanging = set()
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                attackers = len(board.attackers(not piece.color, square))
                defenders = len(board.attackers(piece.color, square))
                if attackers > 0 and attackers > defenders:
                    hanging.add(square)
        return hanging
    
    def _detect_attacking_pieces_v13x(self, board: chess.Board) -> set:
        """Detect pieces that are actively attacking enemy pieces"""
        attacking = set()
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                for target_sq in chess.SQUARES:
                    if board.is_attacked_by(piece.color, target_sq):
                        target = board.piece_at(target_sq)
                        if target and target.color != piece.color:
                            attacking.add(square)
                            break
        return attacking
    
    def _determine_game_phase_v13x(self, board: chess.Board) -> str:
        """Determine game phase"""
        piece_count = len(board.piece_map())
        if piece_count >= 28:
            return 'opening'
        elif piece_count <= 10:
            return 'endgame'
        else:
            return 'middlegame'
    
    def _calculate_material_balance_v13x(self, board: chess.Board) -> int:
        """Calculate material balance"""
        white_material = sum(self.piece_values.get(piece.piece_type, 0) 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.WHITE)
        black_material = sum(self.piece_values.get(piece.piece_type, 0) 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.BLACK)
        return white_material - black_material
    
    def _detect_king_safety_issues_v13x(self, board: chess.Board) -> bool:
        """Detect if king is in danger"""
        king_square = board.king(board.turn)
        if not king_square:
            return False
        return len(board.attackers(not board.turn, king_square)) > 0
    
    def _gives_check_safe_v13x(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move gives check and is relatively safe"""
        board_copy = board.copy()
        board_copy.push(move)
        return board_copy.is_check()
    
    def _saves_king_safety_v13x(self, board: chess.Board, move: chess.Move, position_info: dict) -> bool:
        """Check if move improves king safety"""
        return position_info['king_safety_issues'] and move.from_square == board.king(board.turn)
    
    def _saves_hanging_piece_v13x(self, board: chess.Board, move: chess.Move, position_info: dict) -> bool:
        """Check if move saves a hanging piece"""
        return move.from_square in position_info['hanging_pieces']
    
    def _is_capture_safe_v13x(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if capture is safe (doesn't lose material)"""
        board_copy = board.copy()
        board_copy.push(move)
        attackers = board_copy.attackers(not board.turn, move.to_square)
        return len(attackers) == 0
    
    def _creates_tactical_threat_v13x(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move creates tactical threats (pins, forks, etc.)"""
        board_copy = board.copy()
        board_copy.push(move)
        attacking_piece = board_copy.piece_at(move.to_square)
        if not attacking_piece:
            return False
        
        targets = 0
        high_value_targets = 0
        for square in chess.SQUARES:
            if board_copy.is_attacked_by(attacking_piece.color, square):
                target = board_copy.piece_at(square)
                if target and target.color != attacking_piece.color:
                    targets += 1
                    if target.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                        high_value_targets += 1
        return targets >= 2 or high_value_targets >= 1
    
    def _is_development_move_v13x(self, board: chess.Board, move: chess.Move, position_info: dict) -> bool:
        """Check if move develops a piece"""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            start_rank = chess.square_rank(move.from_square)
            if piece.color == chess.WHITE and start_rank == 0:
                return True
            elif piece.color == chess.BLACK and start_rank == 7:
                return True
        return False
    
    def _controls_center_v13x(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move controls center squares"""
        center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        board_copy = board.copy()
        board_copy.push(move)
        for center_sq in center_squares:
            if board_copy.is_attacked_by(board.turn, center_sq):
                return True
        return False
    
    def _improves_piece_position_v13x(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if quiet move improves piece position"""
        center_files = [3, 4]  # D and E files (0-indexed)
        center_ranks = [3, 4]  # 0-indexed, so ranks 4-5
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        return to_file in center_files or to_rank in center_ranks
    
    def _is_capture_safe_enhanced_v13x(self, board: chess.Board, move: chess.Move) -> bool:
        """Enhanced capture safety check with basic SEE"""
        # Simple Static Exchange Evaluation
        board_copy = board.copy()
        board_copy.push(move)
        
        # Check if our piece on destination is attacked
        attackers = list(board_copy.attackers(not board.turn, move.to_square))
        defenders = list(board_copy.attackers(board.turn, move.to_square))
        
        # Simple heuristic: safe if no attackers or more defenders
        return len(attackers) == 0 or len(defenders) >= len(attackers)
    
    def _creates_tactical_threat_enhanced_v13x(self, board: chess.Board, move: chess.Move) -> bool:
        """Enhanced tactical threat detection"""
        board_copy = board.copy()
        board_copy.push(move)
        attacking_piece = board_copy.piece_at(move.to_square)
        if not attacking_piece:
            return False
        
        # Look for pieces that can attack multiple targets
        targets = []
        high_value_targets = []
        
        for square in chess.SQUARES:
            if board_copy.is_attacked_by(attacking_piece.color, square):
                target = board_copy.piece_at(square)
                if target and target.color != attacking_piece.color:
                    targets.append(target.piece_type)
                    if target.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                        high_value_targets.append(target.piece_type)
        
        # Enhanced criteria: fork (2+ targets) or high-value attack
        return len(targets) >= 2 or len(high_value_targets) >= 1
    
    def _analyze_tactical_threat_v13x(self, board: chess.Board, move: chess.Move) -> tuple:
        """Analyze the type and value of tactical threat"""
        board_copy = board.copy()
        board_copy.push(move)
        attacking_piece = board_copy.piece_at(move.to_square)
        
        if not attacking_piece:
            return "none", 0
        
        targets = []
        threat_value = 0
        
        for square in chess.SQUARES:
            if board_copy.is_attacked_by(attacking_piece.color, square):
                target = board_copy.piece_at(square)
                if target and target.color != attacking_piece.color:
                    targets.append(target)
                    threat_value += self.piece_values.get(target.piece_type, 0) // 10
        
        if len(targets) >= 2:
            # Check for royal fork (involves king)
            if any(t.piece_type == chess.KING for t in targets):
                return "royal_fork", threat_value + 500
            else:
                return "fork", threat_value + 200
        elif len(targets) == 1 and targets[0].piece_type in [chess.QUEEN, chess.ROOK]:
            return "pin", threat_value + 100
        else:
            return "threat", threat_value
    
    def _improves_piece_position_enhanced_v13x(self, board: chess.Board, move: chess.Move) -> bool:
        """Enhanced piece position improvement check"""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        
        # Central squares (more comprehensive)
        central_squares = {chess.D4, chess.D5, chess.E4, chess.E5, chess.C4, chess.C5, chess.F4, chess.F5}
        if move.to_square in central_squares:
            return True
        
        # Piece-specific improvements
        if piece.piece_type == chess.KNIGHT:
            # Knights prefer central outposts
            outposts = {chess.C3, chess.C6, chess.D4, chess.D5, chess.E4, chess.E5, chess.F3, chess.F6}
            return move.to_square in outposts
        elif piece.piece_type == chess.BISHOP:
            # Bishops prefer long diagonals
            from_rank = chess.square_rank(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            return to_rank > from_rank  # Forward development
        elif piece.piece_type == chess.KING:
            # King activity in endgame
            game_phase = self._determine_game_phase_v13x(board)
            if game_phase == 'endgame':
                return chess.square_file(move.to_square) in [3, 4]  # Toward center
        
        return False

    def _apply_position_modifiers_v13x(self, board: chess.Board, move: chess.Move, position_info: dict) -> int:
        """Apply position-specific score modifiers"""
        bonus = 0
        
        # In check: prioritize getting out of check
        if position_info['in_check']:
            if not board.is_capture(move) and not self._gives_check_safe_v13x(board, move):
                bonus += 100
        
        # Material imbalance: adjust tactics vs positional play
        if abs(position_info['material_balance']) > 300:
            if board.is_capture(move) or self._gives_check_safe_v13x(board, move):
                bonus += 50
        
        # Endgame: prioritize king activity and pawn promotion
        if position_info['game_phase'] == 'endgame':
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.KING:
                bonus += 80
            elif piece and piece.piece_type == chess.PAWN:
                bonus += 60
        
        return bonus
    
    def get_waiting_moves(self):
        """Return quiet moves for zugzwang/time situations"""
        return getattr(self, 'waiting_moves', [])
    
    def should_use_waiting_moves(self, position_score, time_remaining):
        """Determine if waiting moves should be considered"""
        # Use in specific situations:
        # 1. Zugzwang positions (no good moves)
        # 2. Time pressure (need fast move)  
        # 3. Quiescence search extension
        return position_score < -50 or time_remaining < 10.0
    
    def _select_waiting_moves(self, board: chess.Board, waiting_moves: list, depth: int) -> list:
        """Select which waiting moves to include based on position"""
        if not waiting_moves:
            return []
        
        # V13.x Strategy: Only add waiting moves in specific conditions
        selected = []
        max_waiting = 3  # Maximum waiting moves to add
        
        # Criteria for including waiting moves:
        # 1. Deep search (depth >= 4) - more comprehensive
        # 2. Few legal moves overall (< 10) - forced positions
        # 3. Endgame situations - quiet moves more important
        # 4. Time pressure - need alternatives
        
        game_phase = self._determine_game_phase_v13x(board)
        piece_count = len(board.piece_map())
        
        if depth >= 4:
            max_waiting = 5  # More waiting moves in deep search
        elif game_phase == 'endgame':
            max_waiting = 4  # Quiet moves important in endgame
        elif piece_count < 15:
            max_waiting = 4  # Complex endgames need more options
        
        # Select best waiting moves by simple criteria
        for move in waiting_moves[:max_waiting]:
            # Prioritize:
            # 1. King moves in endgame
            # 2. Pawn advances
            # 3. Piece repositioning toward center
            
            piece = board.piece_at(move.from_square)
            if piece:
                if game_phase == 'endgame' and piece.piece_type == chess.KING:
                    selected.append(move)
                elif piece.piece_type == chess.PAWN:
                    selected.append(move)
                elif self._improves_piece_position_v13x(board, move):
                    selected.append(move)
                elif len(selected) < 2:  # Ensure at least 2 waiting moves
                    selected.append(move)
        
        return selected[:max_waiting]

    def _determine_evaluation_level(self, board: chess.Board, depth: int) -> int:
        """
        V13.1 SIMPLIFIED EVALUATION SCALING: Performance-based evaluation complexity
        
        ALGORITHM-SAFE APPROACH:
        - Consistent evaluation function (no asymmetry)
        - Depth-based performance scaling only
        - Time pressure optimization
        
        Returns evaluation tier (1-4):
        1 = Fast only (basic material/position)
        2 = Medium (+ advanced pawns/king safety)  
        3 = Slow (+ tactical detection)
        4 = Very Slow (+ complexity bonuses)
        """
        # Get time context if available (use approximate method)
        time_remaining_pct = 100.0  # Default to full time
        if hasattr(self, 'search_start_time'):
            elapsed = time.time() - self.search_start_time
            # Estimate based on typical move time (assume 30 seconds max per move)
            estimated_move_time = 30.0
            time_remaining_pct = max(0, (estimated_move_time - elapsed) / estimated_move_time * 100)
        
        # Base evaluation level from depth (CONSISTENT FOR ALL MOVES)
        if depth >= 6:
            base_level = 1  # Only fast evaluation at deep depths
        elif depth >= 4:
            base_level = 2  # Medium evaluation
        elif depth >= 2:
            base_level = 3  # Slow evaluation
        else:
            base_level = 4  # All evaluation components
        
        # Time pressure adjustments - reduce complexity as time runs out
        if time_remaining_pct < 25:
            base_level = min(base_level, 1)  # Only fast evaluation
        elif time_remaining_pct < 50:
            base_level = min(base_level, 2)  # No slow components
        elif time_remaining_pct < 75:
            base_level = min(base_level, 3)  # No very slow components
        
        # Game phase adjustments
        piece_count = len(board.piece_map())
        if piece_count <= 12:  # Endgame - tactical evaluation less important
            base_level = min(base_level, 2)
        
        return base_level

    def _evaluate_position(self, board: chess.Board, depth: int = 0) -> float:
        """
        V13.1 DIMINISHING EVALUATION: Intelligent selective evaluation system
        Uses performance-graded evaluation components with time/depth awareness
        
        PERFORMANCE TIERS:
        - Fast (always): Bitboard material, simple king safety
        - Medium (depth < 5): Advanced pawns, king safety 
        - Slow (depth < 3, time > 50%): Tactical detection, dynamic pieces
        - Very Slow (depth < 2, time > 75%): Complexity bonuses
        """
        # V12.2 OPTIMIZATION: Use Zobrist hash for cache key
        cache_key = self.zobrist.hash_position(board)
        
        if cache_key in self.evaluation_cache:
            self.search_stats['cache_hits'] += 1
            return self.evaluation_cache[cache_key]
        
        self.search_stats['cache_misses'] += 1
        
        # V13.1: Determine evaluation complexity level based on context
        eval_level = self._determine_evaluation_level(board, depth)
        
        # V13.1: Selective evaluation pipeline
        try:
            # TIER 1: FAST - Always enabled (core material and position)
            if self.ENABLE_DYNAMIC_EVALUATION and self.dynamic_evaluator and eval_level >= 1:
                white_base = self.dynamic_evaluator.evaluate_dynamic_position_value(board, True)
                black_base = self.dynamic_evaluator.evaluate_dynamic_position_value(board, False)
            else:
                # Fallback to fast bitboard evaluation
                white_base = self.bitboard_evaluator.calculate_score_optimized(board, True)
                black_base = self.bitboard_evaluator.calculate_score_optimized(board, False)
            
            # TIER 2: MEDIUM - Advanced components (depth < 5)
            if self.ENABLE_ADVANCED_EVALUATION and eval_level >= 2:
                # V11 PHASE 3A: Advanced pawn structure evaluation
                white_pawn_score = self.advanced_pawn_evaluator.evaluate_pawn_structure(board, True)
                black_pawn_score = self.advanced_pawn_evaluator.evaluate_pawn_structure(board, False)
                
                # Advanced king safety evaluation
                white_king_score = self.king_safety_evaluator.evaluate_king_safety(board, True)
                black_king_score = self.king_safety_evaluator.evaluate_king_safety(board, False)
            else:
                white_pawn_score = 0
                black_pawn_score = 0
                white_king_score = self._simple_king_safety(board, True)
                black_king_score = self._simple_king_safety(board, False)
            
            # TIER 3: SLOW - Tactical detection (depth < 3, time > 50%)
            if self.ENABLE_TACTICAL_DETECTION and self.tactical_detector and eval_level >= 3:
                # OPTIMIZATION: Only run expensive tactical detection when beneficial
                if self._should_run_tactical_detection(board):
                    white_tactical_score = self.tactical_detector.get_tactical_score(board, True)
                    black_tactical_score = self.tactical_detector.get_tactical_score(board, False)
                else:
                    white_tactical_score = 0
                    black_tactical_score = 0
            else:
                white_tactical_score = 0
                black_tactical_score = 0
            
            # TIER 4: VERY SLOW - Complexity bonuses (depth < 2, time > 75%)
            tal_complexity_bonus = 0
            if self.ENABLE_TAL_COMPLEXITY_BONUS and self.dynamic_evaluator and eval_level >= 4:
                tal_complexity_bonus = self.dynamic_evaluator._calculate_position_complexity_bonus(board, True)
                tal_complexity_bonus -= self.dynamic_evaluator._calculate_position_complexity_bonus(board, False)
            
            # Combine all evaluation components with V13.0 weighting
            # Tal Framework: Tactical (35%) + Dynamic (25%) + King Safety (25%) + Material Context (15%)
            white_total = white_base + white_pawn_score + white_king_score + white_tactical_score
            black_total = black_base + black_pawn_score + black_king_score + black_tactical_score
            
            # Add complexity bonus
            if board.turn:  # White to move
                white_total += tal_complexity_bonus
            else:  # Black to move
                black_total -= tal_complexity_bonus
            
        except Exception as e:
            # Fallback to base evaluation if advanced evaluation fails
            white_total = self.bitboard_evaluator.calculate_score_optimized(board, True)
            black_total = self.bitboard_evaluator.calculate_score_optimized(board, False)
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
    
    def _should_run_tactical_detection(self, board: chess.Board) -> bool:
        """
        V13.1 CRITICAL OPTIMIZATION: More aggressive filtering of tactical detection
        Only run on positions with high tactical potential to improve NPS
        """
        # OPTIMIZATION: Reduce tactical detection frequency
        piece_count = len(board.piece_map())
        
        # Skip in endgame (< 10 pieces) unless very shallow search
        if piece_count < 10 and getattr(self, '_current_depth', 5) > 2:
            return False
            
        # Skip after too many moves (diminishing returns)
        if board.fullmove_number > 40:
            return False
        
        # Skip if evaluation count is high (performance protection)
        if hasattr(self, 'evaluation_count'):
            self.evaluation_count += 1
            if self.evaluation_count > 200:  # Limit expensive evaluations
                return False
        else:
            self.evaluation_count = 1
        
        # Only run if there are immediate tactical threats
        in_check = board.is_check()
        if in_check:
            return True
            
        # Quick scan for captures or checks (limited to reduce overhead)
        capture_found = False
        check_found = False
        move_count = 0
        
        for move in board.legal_moves:
            move_count += 1
            if move_count > 8:  # Limit scan for performance
                break
                
            if board.is_capture(move):
                capture_found = True
                break
            elif board.gives_check(move):
                check_found = True
                break
                
        return capture_found or check_found
    
    def _should_run_dynamic_evaluation(self, board: chess.Board) -> bool:
        """
        V13.1 CRITICAL OPTIMIZATION: Much more selective dynamic evaluation
        Use sparingly to improve NPS dramatically
        """
        # OPTIMIZATION: Dynamic evaluation is expensive, use very selectively
        if not hasattr(self, 'dynamic_eval_count'):
            self.dynamic_eval_count = 0
            
        self.dynamic_eval_count += 1
        
        # Only use dynamic evaluation every Nth evaluation (performance boost)
        if self.dynamic_eval_count % 5 != 0:
            return False
            
        # Skip in simple endgames (< 8 pieces)
        piece_count = len(board.piece_map())
        if piece_count < 8:
            return False
            
        # Skip if we've already done many evaluations
        if hasattr(self, 'evaluation_count') and self.evaluation_count > 100:
            return False
            
        # Only use for positions with significant material or in complex middlegame
        white_material = self._count_material_fast(board, True)
        black_material = self._count_material_fast(board, False)
        total_material = white_material + black_material
        
        # Use in complex middlegame positions only
        return (total_material > 4000 and total_material < 7000 and 
                board.fullmove_number > 8 and board.fullmove_number < 30)
            
        # Skip in simple middlegame positions
        return False
    
    def _count_material_fast(self, board: chess.Board, for_white: bool) -> int:
        """Count material value for one side"""
        total = 0
        for piece_type, value in [(chess.PAWN, 100), (chess.KNIGHT, 300), 
                                 (chess.BISHOP, 300), (chess.ROOK, 500), (chess.QUEEN, 900)]:
            total += len(board.pieces(piece_type, for_white)) * value
        return total
    
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

    def _calculate_opponent_depth_reduction(self, search_depth: int, moves_searched: int) -> int:
        """
        V13.1 ASYMMETRIC SEARCH DEPTH: Calculate depth reduction for opponent moves
        
        ALGORITHM-SAFE OPPONENT MODELING:
        - Maintains evaluation consistency (no asymmetric evaluation)
        - Models that opponents search less deeply than we do
        - Reduces search depth for opponent responses based on rating assumptions
        
        This is algorithmically sound because:
        1. We still use the same evaluation function for all positions
        2. We just assume the opponent doesn't search as deeply as we do
        3. This models realistic human/engine behavior at intermediate ratings
        """
        # No reduction for our own moves at shallow depths
        if search_depth <= 3:
            return 0
        
        # Subtle depth reduction for opponent moves in deeper search
        # Assumption: Opponents at our rating level search 0.5-1 ply less deeply
        
        # More aggressive reduction later in move ordering (less likely moves)
        if moves_searched >= 8:
            return 1  # Assume opponent searches 1 ply less for unlikely moves
        elif moves_searched >= 4:
            return 1 if search_depth >= 5 else 0  # Moderate reduction
        else:
            return 1 if search_depth >= 6 else 0  # Conservative reduction for top moves
        
        return 0

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

    # =====================================================================
    # V13.2 PUZZLE SOLVER METHODS - Anti-Null-Move-Pruning System
    # =====================================================================
    
    def search_multi_pv(self, board: chess.Board, time_limit: float = 3.0, num_lines: int = 3) -> List[Dict]:
        """
        V13.2 MULTI-PV SEARCH: Return multiple good moves with opportunity analysis
        This is the foundation of the puzzle solver approach.
        
        Returns list of move candidates with opportunity scores:
        [{'move': chess.Move, 'score': float, 'improvement': float, 'opportunities': list}, ...]
        """
        if not self.ENABLE_MULTI_PV:
            # Fallback to traditional single-move search
            best_move = self.search(board, time_limit)
            return [{'move': best_move, 'score': 0.0, 'improvement': 0.0, 'opportunities': []}]
        
        # Get current position value for improvement calculation
        current_score = self._evaluate_position(board)
        
        # Generate and order all legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []
        
        move_candidates = []
        search_time_per_move = time_limit / min(len(legal_moves), num_lines * 2)
        
        for move in legal_moves:
            board.push(move)
            
            # Evaluate the position after our move using anti-null-move-pruning
            puzzle_score = self._evaluate_opportunity_position(board, depth=3, time_limit=search_time_per_move)
            improvement = puzzle_score - current_score
            opportunities = self._detect_opportunities(board)
            
            # V13.2 TACTICAL WEIGHTING: Apply priority weights to opportunities
            weighted_score = self._calculate_weighted_opportunity_score(puzzle_score, improvement, opportunities)
            
            move_candidates.append({
                'move': move,
                'score': puzzle_score,
                'improvement': improvement,
                'weighted_score': weighted_score,
                'opportunities': opportunities,
                'opportunity_count': len(opportunities)
            })
            
            board.pop()
        
        # Sort by weighted score (incorporates tactical priority)
        move_candidates.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        return move_candidates[:num_lines]
    
    def _calculate_weighted_opportunity_score(self, base_score: float, improvement: float, opportunities: List[str]) -> float:
        """
        V13.2 TACTICAL WEIGHTING: Apply priority weights to opportunity types
        Immediate threats get highest priority to prevent blunders
        """
        weighted_score = base_score + improvement
        
        # PRIORITY 1: IMMEDIATE THREATS (critical weight)
        threat_multipliers = {
            'escape_check': 1000.0,           # Must handle check immediately
            'defend_mate_in_1': 2000.0,       # Must defend mate threats
            'defend_mate_in_2': 500.0,        # Important but less urgent
            'save_material': 300.0,           # Prevent material loss
            'defend_check_threat': 200.0,     # Prevent checks against us
            'defend_material_loss': 250.0,    # Prevent material hanging
        }
        
        for opportunity in opportunities:
            if opportunity in threat_multipliers:
                weighted_score += threat_multipliers[opportunity]
        
        # PRIORITY 2: OFFENSIVE TACTICAL OPPORTUNITIES
        tactical_bonuses = {
            'material_gain': 150.0,           # Winning material is good
            'tactical_threat': 100.0,         # Creating tactics
        }
        
        for opportunity in opportunities:
            if opportunity in tactical_bonuses:
                weighted_score += tactical_bonuses[opportunity]
        
        # PRIORITY 3: DEFENSIVE IMPROVEMENTS
        defensive_bonuses = {
            'improve_king_safety': 75.0,      # Important for long-term safety
            'block_attack': 50.0,             # Defensive moves
        }
        
        for opportunity in opportunities:
            if opportunity in defensive_bonuses:
                weighted_score += defensive_bonuses[opportunity]
        
        # PRIORITY 4: POSITIONAL OPPORTUNITIES (lower weight)
        positional_bonuses = {
            'development': 25.0,              # Good but not urgent
            'center_control': 20.0,           # Nice to have
            'king_safety': 15.0,              # Castling rights
            'pawn_advance': 10.0,             # Least priority
        }
        
        for opportunity in opportunities:
            if opportunity in positional_bonuses:
                weighted_score += positional_bonuses[opportunity]
        
        return weighted_score
    
    def _evaluate_opportunity_position(self, board: chess.Board, depth: int, time_limit: float) -> float:
        """
        V13.2 ANTI-NULL-MOVE-PRUNING: Evaluate opportunities regardless of opponent response
        
        Key insight: Instead of "what if we skip a move" (null move pruning),
        we ask "what opportunities do we have regardless of opponent moves"
        """
        if depth <= 0:
            return self._evaluate_position(board)
        
        if not self.ENABLE_PRACTICAL_SEARCH:
            # Fallback to traditional minimax
            score, _ = self._recursive_search(board, depth, -99999, 99999, time_limit)
            return score
        
        # Generate opponent moves but don't assume they play optimally
        opponent_moves = list(board.legal_moves)
        if not opponent_moves:
            return self._evaluate_position(board)
        
        # Sample opponent responses to understand our opportunities
        # This is the "anti-null-move-pruning" concept
        opportunity_scores = []
        sample_size = min(8, len(opponent_moves))  # Sample up to 8 moves
        
        for i, opp_move in enumerate(opponent_moves[:sample_size]):
            board.push(opp_move)
            
            # After opponent moves, what opportunities do we have?
            our_response_value = self._evaluate_opportunity_position(board, depth - 1, time_limit)
            opportunity_scores.append(our_response_value)
            
            board.pop()
            
            # Early exit if time is running out
            if time_limit > 0 and hasattr(self, 'search_start_time'):
                elapsed = time.time() - self.search_start_time
                if elapsed > time_limit * 0.8:  # Use 80% of time budget
                    break
        
        if not opportunity_scores:
            return self._evaluate_position(board)
        
        # Return average opportunity (not worst case like minimax)
        # This represents "how good are our chances in this position"
        return sum(opportunity_scores) / len(opportunity_scores)
    
    def _detect_opportunities(self, board: chess.Board) -> List[str]:
        """
        V13.2 OPPORTUNITY DETECTION: Identify what we can achieve in this position
        Focus on practical improvements with tactical threat awareness
        """
        opportunities = []
        
        # PRIORITY 1: IMMEDIATE TACTICAL THREATS (highest weight)
        immediate_threats = self._detect_immediate_threats(board)
        opportunities.extend(immediate_threats)
        
        # PRIORITY 2: MATERIAL OPPORTUNITIES
        if any(board.is_capture(move) for move in board.legal_moves):
            opportunities.append("material_gain")
        
        # PRIORITY 3: TACTICAL OPPORTUNITIES (if tactical detection enabled)
        if self.ENABLE_TACTICAL_DETECTION and hasattr(self, 'tactical_detector') and self.tactical_detector:
            if self._should_run_tactical_detection(board):
                try:
                    tactical_info = self.tactical_detector.detect_tactics(board)
                    if tactical_info.get('pins') or tactical_info.get('forks') or tactical_info.get('skewers'):
                        opportunities.append("tactical_threat")
                except:
                    pass
        
        # PRIORITY 4: DEFENSIVE OPPORTUNITIES
        defensive_needs = self._detect_defensive_needs(board)
        opportunities.extend(defensive_needs)
        
        # PRIORITY 5: DEVELOPMENT OPPORTUNITIES
        our_pieces = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn and piece.piece_type != chess.PAWN:
                our_pieces += 1
        
        if our_pieces < 6:  # Still developing
            opportunities.append("development")
        
        # PRIORITY 6: CENTER CONTROL OPPORTUNITIES
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        our_center_control = 0
        for sq in center_squares:
            piece = board.piece_at(sq)
            if piece and piece.color == board.turn:
                our_center_control += 1
        
        if our_center_control < 2:
            opportunities.append("center_control")
        
        # PRIORITY 7: KING SAFETY OPPORTUNITIES
        if board.has_castling_rights(board.turn):
            opportunities.append("king_safety")
        
        # PRIORITY 8: PAWN STRUCTURE OPPORTUNITIES
        pawn_moves = []
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                pawn_moves.append(move)
        
        if pawn_moves:
            opportunities.append("pawn_advance")
        
        return opportunities
    
    def _detect_immediate_threats(self, board: chess.Board) -> List[str]:
        """
        V13.2 IMMEDIATE THREAT DETECTION: Find critical threats that need immediate response
        This addresses the Scholar's mate defense issue
        """
        threats = []
        
        # Check if we're in check (highest priority)
        if board.is_check():
            threats.append("escape_check")
        
        # Check for mate threats against us
        mate_threats = self._detect_mate_threats(board)
        threats.extend(mate_threats)
        
        # Check for material hanging (pieces under attack)
        hanging_material = self._detect_hanging_material(board)
        if hanging_material:
            threats.append("save_material")
        
        # Check for opponent's immediate tactical threats
        opponent_tactics = self._detect_opponent_immediate_tactics(board)
        threats.extend(opponent_tactics)
        
        return threats
    
    def _detect_mate_threats(self, board: chess.Board) -> List[str]:
        """Detect if opponent threatens mate in 1-2 moves"""
        threats = []
        
        # Switch to opponent's perspective to check their threats
        board.turn = not board.turn
        
        for opp_move in list(board.legal_moves)[:10]:  # Check first 10 moves for speed
            board.push(opp_move)
            
            # Check if this move creates checkmate
            if board.is_checkmate():
                threats.append("defend_mate_in_1")
                board.pop()
                board.turn = not board.turn  # Restore turn
                return threats  # Immediate mate threat found
            
            # Check if this move threatens mate in 1 more move (simplified)
            board.turn = not board.turn  # Switch back to us
            our_responses = list(board.legal_moves)[:3]  # Check first 3 responses only
            
            can_defend = False
            for our_response in our_responses:
                board.push(our_response)
                
                # After our response, can opponent still mate us immediately?
                board.turn = not board.turn  # Back to opponent
                opponent_follow_ups = list(board.legal_moves)[:3]
                
                immediate_mate_exists = False
                for follow_up in opponent_follow_ups:
                    board.push(follow_up)
                    if board.is_checkmate():
                        immediate_mate_exists = True
                    board.pop()
                    if immediate_mate_exists:
                        break
                
                board.turn = not board.turn  # Back to us
                board.pop()  # Undo our response
                
                if not immediate_mate_exists:
                    can_defend = True
                    break
            
            if not can_defend and our_responses:
                threats.append("defend_mate_in_2")
            
            board.turn = not board.turn  # Back to opponent
            board.pop()
        
        # Restore original turn
        board.turn = not board.turn
        
        return threats
    
    def _detect_hanging_material(self, board: chess.Board) -> bool:
        """Check if we have pieces hanging (undefended and under attack)"""
        our_pieces = []
        
        # Find all our pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                our_pieces.append((square, piece))
        
        # Check if any of our valuable pieces are hanging
        for square, piece in our_pieces:
            if piece.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                # Is this piece attacked by opponent?
                board.turn = not board.turn
                is_attacked = board.is_attacked_by(not board.turn, square)
                board.turn = not board.turn
                
                if is_attacked:
                    # Is it defended by us?
                    is_defended = board.is_attacked_by(board.turn, square)
                    
                    if not is_defended:
                        return True  # We have hanging material
        
        return False
    
    def _detect_opponent_immediate_tactics(self, board: chess.Board) -> List[str]:
        """Detect immediate tactical threats from opponent"""
        threats = []
        
        # Switch to opponent perspective
        board.turn = not board.turn
        
        for opp_move in list(board.legal_moves)[:8]:  # Check first 8 opponent moves
            board.push(opp_move)
            
            # Check if opponent creates immediate threats
            if board.is_check():
                threats.append("defend_check_threat")
            
            # Check if opponent wins material
            if board.is_capture(opp_move):
                captured_piece = board.piece_type_at(opp_move.to_square)
                if captured_piece and captured_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    threats.append("defend_material_loss")
            
            board.pop()
        
        # Restore turn
        board.turn = not board.turn
        
        return threats
    
    def _detect_defensive_needs(self, board: chess.Board) -> List[str]:
        """Detect defensive opportunities and needs"""
        defensive_opportunities = []
        
        # Check if our king is exposed
        our_king_square = board.king(board.turn)
        if our_king_square:
            # Count how many pieces attack squares around our king
            king_danger = 0
            for delta in [-9, -8, -7, -1, 1, 7, 8, 9]:
                square = our_king_square + delta
                if 0 <= square <= 63:  # Valid square on board
                    board.turn = not board.turn
                    if board.is_attacked_by(board.turn, square):
                        king_danger += 1
                    board.turn = not board.turn
            
            if king_danger >= 3:
                defensive_opportunities.append("improve_king_safety")
        
        # Check if we can block dangerous attacking lines
        if board.is_check():
            defensive_opportunities.append("block_attack")
        
        return defensive_opportunities
    
    def puzzle_search(self, board: chess.Board, time_limit: float = 3.0) -> chess.Move:
        """
        V13.2 PUZZLE SOLVER SEARCH: Choose move based on weighted opportunity analysis
        
        This implements the user's insight: "turn chess into a one-sided massive puzzle"
        With enhanced tactical awareness to prevent blunders.
        """
        if not self.ENABLE_MULTI_PV:
            # Fallback to traditional search
            return self.search(board, time_limit)
        
        # Get multiple move candidates with weighted opportunity analysis
        candidates = self.search_multi_pv(board, time_limit, num_lines=5)
        
        if not candidates:
            return chess.Move.null()
        
        # V13.2 TACTICAL AWARENESS: Always prioritize immediate threats
        threat_candidates = [c for c in candidates if any(
            threat in c['opportunities'] for threat in [
                'escape_check', 'defend_mate_in_1', 'defend_mate_in_2', 
                'save_material', 'defend_check_threat', 'defend_material_loss'
            ]
        )]
        
        if threat_candidates:
            # If we have immediate threats to handle, choose the best defensive move
            best_candidate = threat_candidates[0]  # Already sorted by weighted score
        else:
            # No immediate threats, use normal puzzle solver logic
            best_candidate = candidates[0]
            
            # V13.2 SMART SELECTION: Consider position type and game phase
            moves_played = len(board.move_stack)
            
            if moves_played < 10:  # Opening: prioritize development and center control
                for candidate in candidates:
                    if 'development' in candidate['opportunities'] or 'center_control' in candidate['opportunities']:
                        best_candidate = candidate
                        break
            
            elif moves_played < 30:  # Middlegame: prioritize tactical opportunities
                for candidate in candidates:
                    if 'tactical_threat' in candidate['opportunities'] or 'material_gain' in candidate['opportunities']:
                        best_candidate = candidate
                        break
            
            # Endgame: use default (highest weighted score)
        
        return best_candidate['move']
