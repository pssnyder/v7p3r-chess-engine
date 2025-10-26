#!/usr/bin/env python3
"""
V7P3R Chess Engine v14.2 - Performance Optimizations & Game Phase Detection

Performance-focused build removing V14.1 overhead while adding advanced optimizations.
Built on V14.0 stability with game phase detection and enhanced quiescence search.

ARCHITECTURE:
- Phase 1: Core search (alpha-beta, TT, iterative deepening)
- Phase 2: Performance evaluation (game phase detection, cached values)  
- Phase 3: Optimized move ordering without expensive threat analysis

V14.2 OPTIMIZATIONS:
- REMOVED: Expensive per-move threat detection causing regression
- NEW: Game phase detection (opening/middlegame/endgame) 
- NEW: Enhanced quiescence search with simplified heuristics
- NEW: Advanced time management for consistent 10-ply depth
- NEW: Search depth monitoring and performance profiling
- Cached dynamic bishop valuation for efficiency
- Streamlined move ordering focused on proven heuristics

VERSION LINEAGE:
- v14.2: Performance optimizations - removed overhead, added game phase detection
- v14.1: Enhanced move ordering with threat detection (REGRESSION - too expensive)
- v14.0: Consolidated performance build with unified evaluators
- v12.6: Stable tournament baseline (71.6% score)
- v12.5: Intelligent nudge system experiments
- v12.4: Enhanced castling with balanced evaluation
- v12.2: Performance optimized with tactical regression
- v12.0: Clean evolution with proven improvements only
- v11.x: Experimental variants (lessons learned)
- v10.8: Recovery baseline (19.5/30 tournament points)

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
    """V7P3R Chess Engine v14.2 - Performance Optimizations & Game Phase Detection"""
    
    def __init__(self):
        # Basic configuration
        # Base piece values (V14.2: Cached dynamic bishop valuation)
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300, 
            chess.BISHOP: 300,  # Base value - cached dynamic calculation
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0  # King safety handled separately
        }
        
        # V14.2: Performance optimizations and caching
        self.bishop_value_cache = {}  # Cache for dynamic bishop values
        self.game_phase_cache = {}    # Cache for game phase detection
        self.search_depth_achieved = {}  # Track actual search depth per move
        
        # Search configuration
        self.default_depth = 6
        self.nodes_searched = 0
        
        # Evaluation components - V14.0: Consolidated bitboard evaluation
        self.bitboard_evaluator = V7P3RScoringCalculationBitboard(self.piece_values, enable_nudges=False)
        
        # Simple evaluation cache for speed
        self.evaluation_cache = {}  # position_hash -> evaluation
        
        # Advanced search infrastructure
        self.transposition_table: Dict[int, TranspositionEntry] = {}
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        self.zobrist = ZobristHashing()
        
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
            'average_depth_achieved': 0.0,
            'game_phase_switches': 0,
        }
        
        # PV Following System - V10 OPTIMIZATION
        self.pv_tracker = PVTracker()
    
    def search(self, board: chess.Board, time_limit: float = 3.0, depth: Optional[int] = None, 
               alpha: float = -99999, beta: float = 99999, is_root: bool = True) -> chess.Move:
        """
        V14.2 ADVANCED SEARCH - Enhanced time management and performance monitoring:
        - Iterative deepening with stable best move handling (root level)
        - Alpha-beta pruning with negamax framework (recursive level)  
        - Transposition table with Zobrist hashing
        - Killer moves and history heuristic
        - Streamlined move ordering for performance
        - Advanced time management for consistent 10-ply depth
        - Game phase-aware evaluation and search extension
        - Search depth monitoring and performance profiling
        """
        
        # ROOT LEVEL: Enhanced iterative deepening with advanced time management
        if is_root:
            self.nodes_searched = 0
            self.search_start_time = time.time()
            
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return chess.Move.null()

            # PV FOLLOWING OPTIMIZATION - check if current position triggers instant move
            pv_move = self.pv_tracker.check_position_for_instant_move(board)
            if pv_move:
                return pv_move
            
            # V14.2: Advanced time allocation based on game phase and position complexity
            target_time, max_time = self._calculate_advanced_time_allocation(board, time_limit)
            
            # V14.2: Game phase detection for search strategy
            game_phase = self._detect_game_phase(board)
            
            # V14.2: Target depth based on game phase and time available
            target_depth = self._calculate_target_depth(game_phase, time_limit)
            
            # Iterative deepening with advanced management
            best_move = legal_moves[0]
            best_score = -99999
            depths_completed = []
            
            for current_depth in range(1, target_depth + 1):
                iteration_start = time.time()
                
                # V14.2: Enhanced time checking with phase-aware limits
                elapsed = time.time() - self.search_start_time
                if elapsed > target_time:
                    break
                
                # V14.2: Intelligent iteration prediction
                if current_depth > 2:
                    avg_iteration_time = sum(depths_completed[-2:]) / len(depths_completed[-2:])
                    predicted_time = elapsed + (avg_iteration_time * 2.5)  # Conservative estimate
                    if predicted_time > max_time:
                        break
                
                try:
                    # Store previous best in case iteration fails
                    previous_best = best_move
                    previous_score = best_score
                    
                    # Call recursive search for this depth
                    score, move = self._recursive_search(board, current_depth, -99999, 99999, time_limit)
                    
                    # Track iteration completion time
                    iteration_time = time.time() - iteration_start
                    depths_completed.append(iteration_time)
                    
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
                        
                        # V14.2: Track search depth achieved
                        self.search_depth_achieved[best_move] = current_depth
                        
                        print(f"info depth {current_depth} score cp {int(score)} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {pv_string}")
                        print(f"info string Phase: {game_phase}, Target depth: {target_depth}, Iteration time: {iteration_time:.3f}s")
                        sys.stdout.flush()
                    else:
                        # Restore previous best if iteration failed
                        best_move = previous_best
                        best_score = previous_score
                    
                    # V14.2: Dynamic time management based on position stability
                    if current_depth >= 6:
                        # If we have a stable best move, can potentially search deeper
                        if abs(score - previous_score) < 50:  # Score is stable
                            target_time *= 1.1  # Allow 10% more time for deeper search
                        elif abs(score - previous_score) > 200:  # Score changed significantly
                            target_time *= 0.9  # Reduce time to ensure we have a result
                    
                    elapsed = time.time() - self.search_start_time
                    if elapsed > target_time:
                        break
                        
                except Exception as e:
                    print(f"info string Search interrupted at depth {current_depth}: {e}")
                    break
            
            # V14.2: Update search statistics
            final_depth = len(depths_completed)
            self.search_stats['average_depth_achieved'] = final_depth
            print(f"info string Final depth: {final_depth}, Game phase: {game_phase}, Total time: {time.time() - self.search_start_time:.3f}s")
            
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
        """V14.2 STREAMLINED move ordering - Removed expensive threat detection for performance"""
        if len(moves) <= 2:
            return moves
        
        # Streamlined categories for efficiency
        tt_moves = []
        captures = []
        checks = []
        killers = []
        development = []
        pawn_advances = []
        tactical_moves = []
        quiet_moves = []
        
        # Performance optimization: Pre-create sets for fast lookups
        killer_set = set(self.killer_moves.get_killers(depth))
        
        for move in moves:
            # 1. Transposition table move (highest priority)
            if tt_move and move == tt_move:
                tt_moves.append(move)
                continue
            
            # 2. Captures (sorted by MVV-LVA with cached dynamic values)
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                victim_value = self._get_dynamic_piece_value(board, victim.piece_type, not board.turn) if victim else 0
                attacker = board.piece_at(move.from_square)
                attacker_value = self._get_dynamic_piece_value(board, attacker.piece_type, board.turn) if attacker else 0
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                mvv_lva_score = victim_value * 100 - attacker_value
                
                # Add tactical bonus using bitboards
                tactical_bonus = self.bitboard_evaluator.detect_bitboard_tactics(board, move)
                total_score = mvv_lva_score + tactical_bonus
                
                captures.append((total_score, move))
                continue
            
            # 3. Checks (high priority for tactical play)
            if board.gives_check(move):
                tactical_bonus = self.bitboard_evaluator.detect_bitboard_tactics(board, move)
                checks.append((tactical_bonus, move))
                continue
            
            # 4. Killer moves
            if move in killer_set:
                killers.append(move)
                self.search_stats['killer_hits'] += 1
                continue
            
            # 5. Development and patterns (simplified)
            piece = board.piece_at(move.from_square)
            if piece:
                # Development moves (knights, bishops moving from starting squares)
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    starting_squares = {
                        chess.KNIGHT: [chess.B1, chess.G1, chess.B8, chess.G8],
                        chess.BISHOP: [chess.C1, chess.F1, chess.C8, chess.F8]
                    }
                    if move.from_square in starting_squares.get(piece.piece_type, []):
                        development.append((50.0, move))
                        continue
                
                # Pawn advances
                if piece.piece_type == chess.PAWN:
                    pawn_advances.append((10.0, move))
                    continue
            
            # 6. Remaining moves
            history_score = self.history_heuristic.get_history_score(move)
            tactical_bonus = self.bitboard_evaluator.detect_bitboard_tactics(board, move)
            
            if tactical_bonus > 20.0:  # Significant tactical move
                tactical_moves.append((tactical_bonus + history_score, move))
            else:
                quiet_moves.append((history_score, move))
        
        # Sort all move categories by their scores
        captures.sort(key=lambda x: x[0], reverse=True)
        checks.sort(key=lambda x: x[0], reverse=True)
        development.sort(key=lambda x: x[0], reverse=True)
        pawn_advances.sort(key=lambda x: x[0], reverse=True)
        tactical_moves.sort(key=lambda x: x[0], reverse=True)
        quiet_moves.sort(key=lambda x: x[0], reverse=True)
        
        # V14.2 STREAMLINED ORDER: TT, Captures, Checks, Killers, Development, Pawns, Tactical, Quiet
        ordered = []
        ordered.extend(tt_moves)  # 1. TT move first
        ordered.extend([move for _, move in captures])  # 2. Captures (with cached dynamic values)
        ordered.extend([move for _, move in checks])  # 3. Checks (with tactical bonus)
        ordered.extend(killers)  # 4. Killers
        ordered.extend([move for _, move in development])  # 5. Development moves
        ordered.extend([move for _, move in pawn_advances])  # 6. Pawn advances
        ordered.extend([move for _, move in tactical_moves])  # 7. Tactical patterns
        ordered.extend([move for _, move in quiet_moves])  # 8. Quiet moves
        
        return ordered
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """V12.5 OPTIMIZED: Position evaluation with fast built-in hashing"""
        # V12.5 PERFORMANCE FIX: Use chess library's fast _transposition_key() instead of slow Zobrist hash
        cache_key = board._transposition_key()
        
        if cache_key in self.evaluation_cache:
            self.search_stats['cache_hits'] += 1
            return self.evaluation_cache[cache_key]
        
        self.search_stats['cache_misses'] += 1
        
        # V10 Base bitboard evaluation for material and basic positioning
        white_base = self.bitboard_evaluator.calculate_score_optimized(board, True)
        black_base = self.bitboard_evaluator.calculate_score_optimized(board, False)
        
        # V14.0: Consolidated evaluation components enabled
        try:
            # 2. Advanced pawn structure evaluation 
            white_pawn_score = self.bitboard_evaluator.evaluate_pawn_structure(board, True)
            black_pawn_score = self.bitboard_evaluator.evaluate_pawn_structure(board, False)
            
            # Advanced king safety evaluation
            white_king_score = self.bitboard_evaluator.evaluate_king_safety(board, True)
            black_king_score = self.bitboard_evaluator.evaluate_king_safety(board, False)
            
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
        V14.2 ENHANCED Quiescence search with game phase awareness and simplified heuristics
        Only search captures and checks to avoid horizon effects, but with deeper analysis
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
        
        # V14.2: Enhanced depth limits based on game phase
        game_phase = self._detect_game_phase(board)
        max_quiescence_depth = {
            'opening': 4,    # Shorter quiescence in opening
            'middlegame': 6, # Standard depth for middlegame tactics
            'endgame': 8     # Deeper quiescence for endgame precision
        }.get(game_phase, 6)
        
        # Depth limit reached
        if depth <= -max_quiescence_depth:
            return stand_pat
        
        # Generate and search tactical moves only
        legal_moves = list(board.legal_moves)
        tactical_moves = []
        
        for move in legal_moves:
            # V14.2: Enhanced tactical move selection
            is_capture = board.is_capture(move)
            is_check = board.gives_check(move)
            
            # Always include captures and checks
            if is_capture or is_check:
                tactical_moves.append(move)
                continue
            
            # V14.2: In endgame, also consider pawn promotions and king moves
            if game_phase == 'endgame':
                piece = board.piece_at(move.from_square)
                if piece:
                    # Pawn promotions are critical in endgame
                    if piece.piece_type == chess.PAWN and move.promotion:
                        tactical_moves.append(move)
                    # King activity is important in endgame
                    elif piece.piece_type == chess.KING and depth > -3:  # Only close to leaf nodes
                        tactical_moves.append(move)
        
        # If no tactical moves, return stand pat
        if not tactical_moves:
            return stand_pat
        
        # V14.2: Simplified but efficient tactical move ordering
        capture_scores = []
        for move in tactical_moves:
            score = 0
            
            if board.is_capture(move):
                # Cached dynamic piece values for MVV-LVA
                victim = board.piece_at(move.to_square)
                victim_value = self._get_dynamic_piece_value(board, victim.piece_type, not board.turn) if victim else 0
                attacker = board.piece_at(move.from_square)
                attacker_value = self._get_dynamic_piece_value(board, attacker.piece_type, board.turn) if attacker else 0
                score = victim_value * 100 - attacker_value
                
                # V14.2: Bonus for captures that improve material balance
                if victim_value >= attacker_value:
                    score += 50  # Good trades get priority
            
            elif board.gives_check(move):
                # Check moves get medium priority
                score = 25
                
                # V14.2: Higher priority for checks in endgame
                if game_phase == 'endgame':
                    score += 25
            
            elif move.promotion:
                # Promotions are very valuable
                promoted_piece_value = self.piece_values.get(move.promotion, 0)
                score = promoted_piece_value + 100
            
            capture_scores.append((score, move))
        
        # Sort by score (higher is better)
        capture_scores.sort(key=lambda x: x[0], reverse=True)
        ordered_tactical = [move for _, move in capture_scores]
        
        # V14.2: Delta pruning - skip moves unlikely to improve alpha significantly
        delta_threshold = 200  # Skip moves that can't improve position enough
        if game_phase == 'endgame':
            delta_threshold = 100  # More precise in endgame
        
        # Search tactical moves with simplified pruning
        best_score = stand_pat
        moves_searched = 0
        
        for move in ordered_tactical:
            # V14.2: Delta pruning for efficiency
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                if victim:
                    victim_value = self._get_dynamic_piece_value(board, victim.piece_type, not board.turn)
                    # Skip captures that can't improve alpha enough
                    if stand_pat + victim_value + delta_threshold < alpha:
                        continue
            
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha, depth - 1)
            board.pop()
            
            moves_searched += 1
            
            if score > best_score:
                best_score = score
            
            if score > alpha:
                alpha = score
            
            if alpha >= beta:
                break  # Beta cutoff
            
            # V14.2: Limit search width in deep quiescence for performance
            if depth < -3 and moves_searched >= 8:
                break  # Stop after 8 moves in very deep quiescence
        
        return best_score

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
        """Reset for a new game - V14.2: Enhanced with cache clearing"""
        self.evaluation_cache.clear()
        self.transposition_table.clear()
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        self.nodes_searched = 0
        
        # V14.2: Clear performance optimization caches
        self.bishop_value_cache.clear()
        self.game_phase_cache.clear()
        self.search_depth_achieved.clear()
        
        # Clear PV following data
        self.pv_tracker.clear()
        
        # Reset stats
        for key in self.search_stats:
            if key in ['nodes_per_second', 'cache_hits', 'cache_misses', 'tt_hits', 'tt_stores', 'killer_hits', 'game_phase_switches']:
                self.search_stats[key] = 0
            elif key == 'average_depth_achieved':
                self.search_stats[key] = 0.0

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
    
    def _calculate_advanced_time_allocation(self, board: chess.Board, base_time_limit: float) -> Tuple[float, float]:
        """
        V14.2: Advanced time management for consistent 10-ply depth achievement
        
        Returns: (target_time, max_time)
        """
        # Start with base adaptive allocation
        target_time, max_time = self._calculate_adaptive_time_allocation(board, base_time_limit)
        
        # Game phase specific adjustments
        game_phase = self._detect_game_phase(board)
        phase_factors = {
            'opening': 0.8,    # Faster in opening, rely on book knowledge
            'middlegame': 1.3, # More time for complex tactical analysis
            'endgame': 1.1     # Precise calculation needed but fewer pieces
        }
        time_factor = phase_factors.get(game_phase, 1.0)
        
        # Position complexity analysis
        legal_moves = list(board.legal_moves)
        num_legal_moves = len(legal_moves)
        
        # Critical position detection
        is_critical = (
            board.is_check() or
            num_legal_moves <= 8  # Few legal moves (forced positions)
        )
        
        # Check for major piece captures available
        for move in legal_moves:
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                if victim and self._get_dynamic_piece_value(board, victim.piece_type, not board.turn) >= 500:
                    is_critical = True  # Major piece can be captured
                    break
        
        if is_critical:
            time_factor *= 1.4  # Take significantly more time in critical positions
        
        # Material imbalance consideration
        our_material = self._count_material(board, board.turn)
        their_material = self._count_material(board, not board.turn)
        material_diff = our_material - their_material
        
        if abs(material_diff) > 500:  # Significant material imbalance
            if material_diff < 0:  # We're behind
                time_factor *= 1.2  # Need more precision when behind
            else:  # We're ahead
                time_factor *= 0.85  # Can play faster when ahead
        
        # Calculate enhanced times for deeper search
        enhanced_target = min(target_time * time_factor, base_time_limit * 0.85)
        enhanced_max = min(max_time * time_factor, base_time_limit * 0.95)
        
        return enhanced_target, enhanced_max
    
    def _calculate_target_depth(self, game_phase: str, time_limit: float) -> int:
        """
        V14.2: Calculate target search depth based on game phase and available time
        
        Goal: Achieve consistent 10-ply depth when possible
        """
        base_depths = {
            'opening': 8,     # Standard opening depth
            'middlegame': 10, # Target 10-ply for complex positions
            'endgame': 12     # Can search deeper in endgame with fewer pieces
        }
        
        base_depth = base_depths.get(game_phase, 10)
        
        # Adjust based on available time
        if time_limit >= 10.0:  # Plenty of time
            return min(base_depth + 2, 15)
        elif time_limit >= 5.0:  # Standard time
            return base_depth
        elif time_limit >= 2.0:  # Limited time
            return max(base_depth - 1, 6)
        else:  # Very limited time
            return max(base_depth - 2, 5)
    
    def _count_material(self, board: chess.Board, color: bool) -> int:
        """Count total material for a color (V14.1: Dynamic piece values)"""
        total = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            pieces = board.pieces(piece_type, color)
            piece_value = self._get_dynamic_piece_value(board, piece_type, color)
            total += len(pieces) * piece_value
        return total

    def _get_dynamic_piece_value(self, board: chess.Board, piece_type: int, color: bool) -> int:
        """V14.2: Cached dynamic piece valuation for performance"""
        if piece_type != chess.BISHOP:
            return self.piece_values[piece_type]
        
        # Use cached value if available
        board_hash = hash(str(board.pieces(chess.BISHOP, chess.WHITE)) + str(board.pieces(chess.BISHOP, chess.BLACK)))
        cache_key = (board_hash, color)
        
        if cache_key in self.bishop_value_cache:
            return self.bishop_value_cache[cache_key]
        
        # Calculate dynamic bishop value
        bishops = board.pieces(chess.BISHOP, color)
        bishop_count = len(bishops)
        
        if bishop_count >= 2:
            # Bishop pair bonus: 325 each (two bishops > two knights)
            value = 325
        elif bishop_count == 1:
            # Single bishop penalty: 275 (one bishop < one knight)
            value = 275
        else:
            # No bishops remaining
            value = self.piece_values[piece_type]
        
        # Cache for future use
        self.bishop_value_cache[cache_key] = value
        return value

    def _detect_game_phase(self, board: chess.Board) -> str:
        """V14.2: NEW - Detect game phase for dynamic evaluation
        
        Returns: 'opening', 'middlegame', or 'endgame'
        """
        # Use cached value if available
        material_hash = self._calculate_material_hash(board)
        if material_hash in self.game_phase_cache:
            return self.game_phase_cache[material_hash]
        
        # Count total material (excluding pawns and kings)
        total_material = 0
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                piece_count = len(board.pieces(piece_type, color))
                total_material += piece_count * self.piece_values[piece_type]
        
        # Phase detection based on material and move count
        moves_played = len(board.move_stack)
        
        if moves_played < 8 and total_material >= 5000:  # Early opening with most pieces
            phase = 'opening'
        elif total_material <= 2500:  # Low material remaining
            phase = 'endgame'
        else:
            phase = 'middlegame'
        
        # Cache for future use
        self.game_phase_cache[material_hash] = phase
        return phase

    def _calculate_material_hash(self, board: chess.Board) -> int:
        """Calculate a hash based on material for caching game phases"""
        material_vector = []
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
                material_vector.append(len(board.pieces(piece_type, color)))
        return hash(tuple(material_vector))

    def get_performance_report(self) -> str:
        """V14.2: Generate performance report for analysis"""
        avg_depth = self.search_stats.get('average_depth_achieved', 0)
        total_moves = len(self.search_depth_achieved)
        
        depth_distribution = {}
        for depth in self.search_depth_achieved.values():
            depth_distribution[depth] = depth_distribution.get(depth, 0) + 1
        
        cache_hits = self.search_stats.get('cache_hits', 0)
        cache_misses = self.search_stats.get('cache_misses', 0)
        cache_hit_rate = cache_hits / max(cache_hits + cache_misses, 1)
        
        tt_hits = self.search_stats.get('tt_hits', 0)
        tt_hit_rate = tt_hits / max(self.nodes_searched, 1)
        
        report = f"""
V14.2 Performance Report:
========================
Average Search Depth: {avg_depth:.1f} ply
Total Moves Analyzed: {total_moves}
Nodes Per Second: {self.search_stats.get('nodes_per_second', 0)}
Cache Hit Rate: {cache_hit_rate:.1%}
TT Hit Rate: {tt_hit_rate:.1%}

Depth Distribution:
{chr(10).join(f"  {depth} ply: {count} moves" for depth, count in sorted(depth_distribution.items()))}

Game Phase Switches: {self.search_stats.get('game_phase_switches', 0)}
Bishop Value Cache Size: {len(self.bishop_value_cache)}
Game Phase Cache Size: {len(self.game_phase_cache)}
"""
        return report

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
