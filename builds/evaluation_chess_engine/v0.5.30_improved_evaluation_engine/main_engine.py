# main_engine.py
"""
Main Chess Engine Controller
Integrates evaluation engine, search, and time management
"""

import chess
import time
import threading
from typing import Optional, Callable, Dict, Any, List, Tuple
from evaluation_engine import EvaluationEngine
from time_manager import TimeManager
import random

class ChessEngine:
    def __init__(self):
        self.evaluation_engine = EvaluationEngine(chess.Board(), depth=6)
        self.time_manager = TimeManager()
        self.max_depth = 8
        self.hash_size = 64  # MB
        self.threads = 1
        self.nodes_searched = 0
        self.transposition_table = {}
        self.killer_moves = [[None, None] for _ in range(50)]  # 2 killer moves per ply
        self.history_table = {}

    def new_game(self):
        """Reset engine for new game"""
        self.transposition_table.clear()
        self.killer_moves = [[None, None] for _ in range(50)]
        self.history_table.clear()
        self.nodes_searched = 0

    def set_max_depth(self, depth: int):
        """Set maximum search depth"""
        self.max_depth = max(1, min(depth, 20))
        self.evaluation_engine.depth = self.max_depth

    def set_hash_size(self, size_mb: int):
        """Set transposition table size"""
        self.hash_size = max(1, min(size_mb, 1024))
        # Clear table when resizing
        self.transposition_table.clear()

    def set_threads(self, threads: int):
        """Set number of search threads"""
        self.threads = max(1, min(threads, 8))

    def search(self, board: chess.Board, time_control: Dict[str, Any], 
               stop_callback: Callable[[], bool] = None) -> Optional[chess.Move]:
        """
        Main search function

        Args:
            board: Current position
            time_control: Time control parameters
            stop_callback: Function to check if search should stop

        Returns:
            Best move found
        """
        self.nodes_searched = 0

        # Set up evaluation engine with current board
        self.evaluation_engine.board = board.copy()

        # Allocate time for this move
        allocated_time = self.time_manager.allocate_time(time_control, board)
        self.time_manager.start_timer(allocated_time)

        # Handle special cases
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        if len(legal_moves) == 1:
            return legal_moves[0]  # Only one legal move

        # Determine search depth
        max_depth = self.max_depth
        if time_control.get('depth'):
            max_depth = time_control['depth']

        best_move = None
        best_score = float('-inf')

        try:
            # Iterative deepening search
            for depth in range(1, max_depth + 1):
                if stop_callback and stop_callback():
                    break
                if self.time_manager.should_stop(depth, self.nodes_searched):
                    break

                move, score = self.search_depth(board, depth, stop_callback)

                if move:
                    best_move = move
                    best_score = score

                    # Print UCI info
                    elapsed = self.time_manager.time_elapsed()
                    nps = int(self.nodes_searched / max(elapsed, 0.001))

                    print(f"info depth {depth} score cp {int(score * 100)} "
                          f"nodes {self.nodes_searched} time {int(elapsed * 1000)} "
                          f"nps {nps} pv {move.uci()}")

                # Don't continue if we found a mate
                if abs(score) > 900:  # Mate score threshold
                    break

        except Exception as e:
            print(f"info string Search error: {e}")

        return best_move or legal_moves[0]  # Fallback to first legal move

    def search_depth(self, board: chess.Board, depth: int, 
                    stop_callback: Callable[[], bool] = None) -> Tuple[Optional[chess.Move], float]:
        """
        Search to a specific depth

        Args:
            board: Current position
            depth: Search depth
            stop_callback: Function to check if search should stop

        Returns:
            Tuple of (best_move, best_score)
        """
        alpha = float('-inf')
        beta = float('inf')
        best_move = None
        best_score = float('-inf')

        # Get ordered moves for better alpha-beta pruning
        moves = self.order_moves(board, depth)

        for move in moves:
            if stop_callback and stop_callback():
                break
            if self.time_manager.should_stop(depth, self.nodes_searched):
                break

            board.push(move)
            self.nodes_searched += 1

            # Search the position after this move
            score = -self.negamax(board, depth - 1, -beta, -alpha, stop_callback)

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
                alpha = max(alpha, score)

                # Update killer moves
                if depth < len(self.killer_moves):
                    if self.killer_moves[depth][0] != move:
                        self.killer_moves[depth][1] = self.killer_moves[depth][0]
                        self.killer_moves[depth][0] = move

        return best_move, best_score

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float,
                stop_callback: Callable[[], bool] = None) -> float:
        """
        Negamax search with alpha-beta pruning

        Args:
            board: Current position
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            stop_callback: Function to check if search should stop

        Returns:
            Position evaluation score
        """
        if stop_callback and stop_callback():
            return 0.0
        if self.time_manager.should_stop(depth, self.nodes_searched):
            return 0.0

        # Terminal node evaluation
        if depth == 0:
            return self.quiescence_search(board, alpha, beta, stop_callback)

        # Check for game over
        if board.is_game_over():
            if board.is_checkmate():
                return -1000 + (self.max_depth - depth)  # Prefer faster mates
            else:
                return 0  # Draw

        # Transposition table lookup
        board_hash = chess.polyglot.zobrist_hash(board)
        if board_hash in self.transposition_table:
            entry = self.transposition_table[board_hash]
            if entry['depth'] >= depth:
                if entry['flag'] == 'EXACT':
                    return entry['score']
                elif entry['flag'] == 'LOWERBOUND' and entry['score'] >= beta:
                    return entry['score']
                elif entry['flag'] == 'UPPERBOUND' and entry['score'] <= alpha:
                    return entry['score']

        original_alpha = alpha
        best_score = float('-inf')
        best_move = None

        # Get ordered moves
        moves = self.order_moves(board, depth)

        for move in moves:
            if stop_callback and stop_callback():
                break
            if self.time_manager.should_stop(depth, self.nodes_searched):
                break

            board.push(move)
            self.nodes_searched += 1

            score = -self.negamax(board, depth - 1, -beta, -alpha, stop_callback)

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if alpha >= beta:
                # Update killer moves for cutoff
                if depth < len(self.killer_moves) and not board.is_capture(move):
                    if self.killer_moves[depth][0] != move:
                        self.killer_moves[depth][1] = self.killer_moves[depth][0]
                        self.killer_moves[depth][0] = move
                break  # Beta cutoff

        # Store in transposition table
        flag = 'EXACT'
        if best_score <= original_alpha:
            flag = 'UPPERBOUND'
        elif best_score >= beta:
            flag = 'LOWERBOUND'

        self.transposition_table[board_hash] = {
            'score': best_score,
            'depth': depth,
            'flag': flag,
            'move': best_move
        }

        # Limit transposition table size
        if len(self.transposition_table) > self.hash_size * 1000:
            # Remove random entries to free space
            keys_to_remove = random.sample(list(self.transposition_table.keys()), 
                                         len(self.transposition_table) // 4)
            for key in keys_to_remove:
                del self.transposition_table[key]

        return best_score

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float,
                         stop_callback: Callable[[], bool] = None, depth: int = 0) -> float:
        """
        Quiescence search to avoid horizon effect

        Args:
            board: Current position
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            stop_callback: Function to check if search should stop
            depth: Current quiescence depth

        Returns:
            Position evaluation score
        """
        if stop_callback and stop_callback():
            return 0.0
        if depth > 10:  # Limit quiescence depth
            return self.evaluate_position(board)

        # Stand pat score
        stand_pat = self.evaluate_position(board)

        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Only search captures in quiescence
        captures = [move for move in board.legal_moves if board.is_capture(move)]

        # Order captures by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        captures.sort(key=lambda move: self.mvv_lva_score(board, move), reverse=True)

        for move in captures:
            if stop_callback and stop_callback():
                break

            board.push(move)
            self.nodes_searched += 1

            score = -self.quiescence_search(board, -beta, -alpha, stop_callback, depth + 1)

            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate position using evaluation engine"""
        # Update evaluation engine board
        self.evaluation_engine.board = board.copy()

        # Get evaluation from evaluation engine
        try:
            score = self.evaluation_engine.evaluate_position()
            return score
        except Exception:
            # Fallback to simple material evaluation
            return self.simple_material_evaluation(board)

    def simple_material_evaluation(self, board: chess.Board) -> float:
        """Simple material-based evaluation as fallback"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }

        score = 0
        for square, piece in board.piece_map().items():
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value

        return score if board.turn == chess.WHITE else -score

    def order_moves(self, board: chess.Board, depth: int) -> List[chess.Move]:
        """
        Order moves for better alpha-beta pruning

        Args:
            board: Current position
            depth: Current search depth

        Returns:
            Ordered list of moves
        """
        moves = list(board.legal_moves)
        move_scores = []

        for move in moves:
            score = 0

            # Transposition table move gets highest priority
            board_hash = chess.polyglot.zobrist_hash(board)
            if board_hash in self.transposition_table:
                if self.transposition_table[board_hash].get('move') == move:
                    score += 10000

            # Captures (MVV-LVA)
            if board.is_capture(move):
                score += self.mvv_lva_score(board, move)

            # Killer moves
            if depth < len(self.killer_moves):
                if move == self.killer_moves[depth][0]:
                    score += 900
                elif move == self.killer_moves[depth][1]:
                    score += 800

            # Checks
            board.push(move)
            if board.is_check():
                score += 500
            board.pop()

            # Promotions
            if move.promotion:
                score += 700

            # History heuristic
            move_key = (move.from_square, move.to_square)
            score += self.history_table.get(move_key, 0)

            move_scores.append((move, score))

        # Sort by score (descending)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, score in move_scores]

    def mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        """Most Valuable Victim - Least Valuable Attacker score"""
        piece_values = [0, 1, 3, 3, 5, 9, 10]  # None, P, N, B, R, Q, K

        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)

        if victim is None:
            return 0

        victim_value = piece_values[victim.piece_type]
        attacker_value = piece_values[attacker.piece_type]

        return victim_value * 100 - attacker_value

# Example usage and testing
if __name__ == "__main__":
    import sys

    # Test the engine
    engine = ChessEngine()
    board = chess.Board()

    # Simple time control for testing
    time_control = {'movetime': 1000}  # 1 second per move

    print("Testing Chess Engine...")
    print("Initial position:")
    print(board)
    print()

    # Test a few moves
    for move_num in range(3):
        best_move = engine.search(board, time_control)
        if best_move:
            print(f"Move {move_num + 1}: {best_move.uci()} ({best_move})")
            board.push(best_move)
            print(board)
            print()
        else:
            print("No legal moves found!")
            break

    print("Engine test completed!")
