# v7p3r.py - V7P3R Chess Engine v1.2 (Consolidated from v1.0 and v1.1)

""" V7P3R Evaluation Engine v1.2
Consolidated version of the V7P3R chess AI for C++ conversion reference.
Built from v1.0 and v1.1 foundations with simplified configuration and no complex dependencies.
"""

from __future__ import annotations
import chess
import yaml
import random
import time
from typing import Optional, Callable, Dict, Any, Tuple
from collections import OrderedDict
from engine_utilities.piece_square_tables import PieceSquareTables
from engine_utilities.time_manager import TimeManager
from engine_utilities.opening_book import OpeningBook
from engine_utilities.v7p3r_scoring_calculation import V7P3RScoringCalculation


class LimitedSizeDict(OrderedDict):
    """Simple dictionary with size limit for transposition table."""
    def __init__(self, *args, maxlen=100000, **kwargs):
        self.maxlen = maxlen
        super().__init__(*args, **kwargs)
    
    def __setitem__(self, key, value):
        if key in self:
            # Move existing key to end
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxlen:
            # Remove oldest entry
            oldest = next(iter(self))
            del self[oldest]


class V7P3REvaluationEngine:
    """Main V7P3R Chess Engine class - simplified version based on v1.0 and v1.1."""
    
    def __init__(self, board: chess.Board = chess.Board(), player: chess.Color = chess.WHITE, ai_config=None):
        self.board = board
        self.current_player = player
        self.time_manager = TimeManager()
        self.opening_book = OpeningBook()

        # Search tracking
        self.nodes_searched = 0
        self.transposition_table = LimitedSizeDict(maxlen=100000) 
        self.killer_moves = [[None, None] for _ in range(50)] 
        self.history_table = {}
        self.counter_moves = {}

        # Piece values (in pawns)
        self.piece_values = {
            chess.KING: 0.0,
            chess.QUEEN: 9.0,
            chess.ROOK: 5.0,
            chess.BISHOP: 3.25,
            chess.KNIGHT: 3.0,
            chess.PAWN: 1.0
        }

        # Load configuration from YAML
        try:
            with open("v7p3r.yaml") as f:
                yaml_data = yaml.safe_load(f) or {}
                self.config_data = yaml_data.get('v7p3r', {})
                self.eval_config = yaml_data.get('default_evaluation', {})
        except Exception as e:
            print(f"Warning: Could not load v7p3r.yaml: {e}")
            self.config_data = {}
            self.eval_config = {}

        # Apply configuration or use defaults
        self.ai_config = ai_config or self.config_data
        self.depth = self.ai_config.get('depth', 4)
        self.max_depth = self.ai_config.get('max_depth', 8)
        self.use_pst = self.ai_config.get('pst', True)
        self.pst_weight = self.ai_config.get('pst_weight', 1.2)
        self.move_ordering = self.ai_config.get('move_ordering', True)
        self.quiescence = self.ai_config.get('quiescence', True)
        self.game_phase_awareness = self.ai_config.get('game_phase_awareness', True)
        
        # Initialize components
        self.pst = PieceSquareTables()
        self.scoring_calculator = V7P3RScoringCalculation(
            v7p3r_yaml_config=self.config_data,
            ai_config=self.ai_config,
            piece_values=self.piece_values,
            pst=self.pst
        )
        
        # Game state tracking
        self.endgame_factor = 0.0
        self.reset(self.board)

    def reset(self, board: chess.Board):
        """Reset engine state for new position."""
        self.board = board.copy()
        self.nodes_searched = 0
        self.transposition_table.clear()
        self.killer_moves = [[None, None] for _ in range(50)]
        self.history_table.clear()
        self.counter_moves.clear()
        self._update_game_phase()

    def _update_game_phase(self):
        """Calculate endgame factor based on material count."""
        if not self.game_phase_awareness:
            self.endgame_factor = 0.0
            return
            
        # Count material
        material_count = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                material_count += self.piece_values[piece.piece_type]
        
        # Endgame factor: 0.0 at start (78 points), 1.0 when only kings remain
        start_material = 78.0  # Total starting material minus kings
        self.endgame_factor = max(0.0, min(1.0, (start_material - material_count) / start_material))

    def get_best_move(self, board: chess.Board = None, time_limit: float = None) -> Optional[chess.Move]:
        """Get the best move for the current position."""
        if board:
            self.reset(board)
        
        if not list(self.board.legal_moves):
            return None
            
        # Check opening book first
        book_move = self.opening_book.get_book_move(self.board)
        if book_move and book_move in self.board.legal_moves:
            return book_move
            
        # Set time limit
        if time_limit:
            self.time_manager.set_time_limit(time_limit)
        
        # Iterative deepening search
        best_move = None
        start_time = time.time()
        
        for depth in range(1, self.max_depth + 1):
            if time_limit and (time.time() - start_time) > time_limit * 0.8:
                break
                
            try:
                current_best = self._search_depth(depth)
                if current_best:
                    best_move = current_best
            except KeyboardInterrupt:
                break
                
        return best_move or random.choice(list(self.board.legal_moves))

    def _search_depth(self, depth: int) -> Optional[chess.Move]:
        """Search to specific depth using alpha-beta."""
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Order moves for better alpha-beta pruning
        moves = list(self.board.legal_moves)
        if self.move_ordering:
            moves = self._order_moves(moves, depth)
            
        for move in moves:
            self.board.push(move)
            self.nodes_searched += 1
            
            score = -self._negamax(depth - 1, -beta, -alpha, False)
            
            self.board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
                
            alpha = max(alpha, score)
            if alpha >= beta:
                # Store killer move
                if depth < len(self.killer_moves):
                    if self.killer_moves[depth][0] != move:
                        self.killer_moves[depth][1] = self.killer_moves[depth][0]
                        self.killer_moves[depth][0] = move
                break
                
        return best_move

    def _negamax(self, depth: int, alpha: float, beta: float, is_quiescence: bool = False) -> float:
        """Negamax search with alpha-beta pruning."""
        self.nodes_searched += 1
        
        # Check for immediate termination conditions
        if self.board.is_checkmate():
            return float('-inf')
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.0
            
        # Transposition table lookup
        board_hash = hash(str(self.board.fen()))
        if board_hash in self.transposition_table:
            tt_entry = self.transposition_table[board_hash]
            if tt_entry['depth'] >= depth:
                return tt_entry['score']
        
        # Quiescence search at leaf nodes
        if depth <= 0:
            if self.quiescence and not is_quiescence:
                return self._quiescence_search(alpha, beta)
            else:
                return self._evaluate_position()
        
        # Generate and order moves
        moves = list(self.board.legal_moves)
        if not moves:
            return 0.0
            
        if self.move_ordering:
            moves = self._order_moves(moves, depth)
        
        best_score = float('-inf')
        
        for move in moves:
            self.board.push(move)
            score = -self._negamax(depth - 1, -beta, -alpha, is_quiescence)
            self.board.pop()
            
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            
            if alpha >= beta:
                # Store killer move
                if depth < len(self.killer_moves):
                    if self.killer_moves[depth][0] != move:
                        self.killer_moves[depth][1] = self.killer_moves[depth][0]
                        self.killer_moves[depth][0] = move
                # Update history table
                if move not in self.history_table:
                    self.history_table[move] = 0
                self.history_table[move] += depth * depth
                break
        
        # Store in transposition table
        self.transposition_table[board_hash] = {
            'score': best_score,
            'depth': depth
        }
        
        return best_score

    def _quiescence_search(self, alpha: float, beta: float, depth: int = 4) -> float:
        """Quiescence search for tactical positions."""
        if depth <= 0:
            return self._evaluate_position()
            
        # Stand pat score
        stand_pat = self._evaluate_position()
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        
        # Only consider captures in quiescence search
        captures = [move for move in self.board.legal_moves if self.board.is_capture(move)]
        
        for move in captures:
            self.board.push(move)
            score = -self._quiescence_search(-beta, -alpha, depth - 1)
            self.board.pop()
            
            if score >= beta:
                return beta
            alpha = max(alpha, score)
            
        return alpha

    def _order_moves(self, moves: list, depth: int) -> list:
        """Order moves for better alpha-beta pruning."""
        if not self.move_ordering:
            return moves
            
        scored_moves = []
        
        for move in moves:
            score = 0
            
            # Hash move (from transposition table)
            board_hash = hash(str(self.board.fen()))
            if board_hash in self.transposition_table:
                score += self.eval_config.get('hash_move_bonus', 5000)
            
            # Captures (MVV-LVA)
            if self.board.is_capture(move):
                captured_piece = self.board.piece_at(move.to_square)
                moving_piece = self.board.piece_at(move.from_square)
                if captured_piece and moving_piece:
                    score += (self.piece_values[captured_piece.piece_type] * 100 - 
                             self.piece_values[moving_piece.piece_type] * 10)
                score += self.eval_config.get('capture_move_bonus', 4000)
            
            # Promotions
            if move.promotion:
                score += self.eval_config.get('promotion_move_bonus', 3000)
            
            # Checks
            self.board.push(move)
            if self.board.is_check():
                score += self.eval_config.get('check_move_bonus', 10000)
                if self.board.is_checkmate():
                    score += self.eval_config.get('checkmate_move_bonus', 1000000)
            self.board.pop()
            
            # Killer moves
            if depth < len(self.killer_moves):
                if move == self.killer_moves[depth][0]:
                    score += self.eval_config.get('killer_move_bonus', 2000)
                elif move == self.killer_moves[depth][1]:
                    score += self.eval_config.get('killer_move_bonus', 2000) // 2
            
            # History heuristic
            if move in self.history_table:
                score += min(self.history_table[move], self.eval_config.get('history_move_bonus', 1000))
            
            scored_moves.append((score, move))
        
        # Sort by score (highest first)
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in scored_moves]

    def _evaluate_position(self) -> float:
        """Evaluate the current position."""
        self._update_game_phase()
        return self.scoring_calculator.evaluate_board(self.board, endgame_factor=self.endgame_factor)

    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information."""
        return {
            'name': 'V7P3R Chess Engine',
            'version': '1.2.0',
            'author': 'Pat Snyder',
            'nodes_searched': self.nodes_searched,
            'transposition_table_size': len(self.transposition_table),
            'endgame_factor': self.endgame_factor
        }


# Simple UCI interface for testing
def main():
    """Simple main function for basic testing."""
    engine = V7P3REvaluationEngine()
    board = chess.Board()
    
    print("V7P3R Chess Engine v1.2 - Consolidated Version")
    print("Enter moves in algebraic notation (e.g., e2e4) or 'quit' to exit")
    
    while not board.is_game_over():
        print(f"\nPosition: {board.fen()}")
        print(f"Legal moves: {len(list(board.legal_moves))}")
        
        if board.turn == chess.WHITE:
            # Human move
            move_str = input("Your move: ").strip()
            if move_str.lower() == 'quit':
                break
                
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move!")
                    continue
            except:
                print("Invalid move format!")
                continue
        else:
            # Engine move
            print("Engine thinking...")
            start_time = time.time()
            move = engine.get_best_move(board, time_limit=5.0)
            think_time = time.time() - start_time
            
            if move:
                board.push(move)
                print(f"Engine plays: {move} (thought for {think_time:.2f}s)")
                info = engine.get_engine_info()
                print(f"Nodes searched: {info['nodes_searched']}")
            else:
                print("Engine found no move!")
                break
    
    print(f"\nGame over: {board.result()}")


if __name__ == "__main__":
    main()
