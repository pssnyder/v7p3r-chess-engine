# v7p3r_engine.py

"""V7P3R Engine
This module coordinates the chess engine module handlers for complete engine operation during a game 
or when asked to find a move for any position, such as during puzzle solving.
"""

import chess
import time
from v7p3r_config import V7P3RConfig
from v7p3r_search import SearchController
from v7p3r_scoring import ScoringSystem
from v7p3r_rules import GameRules
from v7p3r_book import OpeningBook

class V7P3REngine:
    def __init__(self, config_file="config.json"):
        # Load configuration
        self.config = V7P3RConfig(config_file)
        engine_config = self.config.get_engine_config()
        
        # Engine identification
        self.name = engine_config.get('name', 'v7p3r')
        self.version = engine_config.get('version', '1.0.0')
        self.engine_id = engine_config.get('engine_id', 'v7p3r_default')
        
        # Initialize components
        self.search = SearchController(self.config)
        self.scoring = ScoringSystem(self.config)
        self.rules = GameRules(self.config)
        self.book = OpeningBook() if engine_config.get('use_opening_book', True) else None
        
        # Move history and statistics
        self.move_history = []
        self.position_history = []
        self.game_stats = {
            'moves_played': 0,
            'book_moves': 0,
            'search_time_total': 0,
            'nodes_searched_total': 0
        }
    
    def find_move(self, board, time_limit=30.0):
        """Find the best move for the current position"""
        start_time = time.time()
        our_color = board.turn
        
        # Validate board state
        if board.is_game_over():
            return None
        
        # Check opening book first
        book_move = None
        if self.book and self.config.is_enabled('engine_config', 'use_opening_book'):
            book_move = self.book.get_book_move(board)
            if book_move:
                self.game_stats['book_moves'] += 1
                return book_move
        
        # Use search to find best move
        best_move = self.search.find_best_move(board, our_color)
        
        # Validate the selected move
        if best_move:
            is_valid, reason = self.rules.validate_move(board, best_move)
            if not is_valid:
                # Fall back to simple search
                print(f"Warning: Selected move invalid ({reason}), using fallback")
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    # Simple evaluation of all legal moves
                    best_score = float('-inf')
                    fallback_move = legal_moves[0]
                    
                    for move in legal_moves:
                        score, _, _ = self.scoring.evaluate_move(board, move, our_color, depth=1)
                        if score > best_score:
                            best_score = score
                            fallback_move = move
                    
                    best_move = fallback_move
        
        # Check time limit
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            print(f"Warning: Move selection took {elapsed_time:.2f}s (limit: {time_limit}s)")
        
        # Update statistics
        search_stats = self.search.get_search_stats()
        self.game_stats['moves_played'] += 1
        self.game_stats['search_time_total'] += search_stats['search_time']
        self.game_stats['nodes_searched_total'] += search_stats['nodes_searched']
        
        return best_move
    
    def make_move(self, board, move):
        """Make a move and update internal state"""
        if move in board.legal_moves:
            # Store position before move
            self.position_history.append(board.fen())
            
            # Make the move
            board.push(move)
            self.move_history.append(move)
            
            return True
        return False
    
    def get_evaluation(self, board):
        """Get static evaluation of current position"""
        our_color = board.turn
        return self.scoring.evaluate_position(board, our_color)
    
    def get_position_analysis(self, board):
        """Get detailed analysis of current position"""
        our_color = board.turn
        guidelines = self.rules.get_position_guidelines(board, our_color)
        
        analysis = {
            'guidelines': guidelines,
            'evaluation': self.get_evaluation(board),
            'in_book': self.book.is_in_book(board) if self.book else False,
            'legal_moves_count': len(list(board.legal_moves))
        }
        
        return analysis
    
    def reset_game(self):
        """Reset engine state for a new game"""
        self.move_history = []
        self.position_history = []
        self.game_stats = {
            'moves_played': 0,
            'book_moves': 0,
            'search_time_total': 0,
            'nodes_searched_total': 0
        }
    
    def get_engine_info(self):
        """Get engine information"""
        book_stats = self.book.get_book_statistics() if self.book else {}
        
        return {
            'name': self.name,
            'version': self.version,
            'engine_id': self.engine_id,
            'search_algorithm': self.config.get_setting('engine_config', 'search_algorithm'),
            'max_depth': self.config.get_setting('engine_config', 'depth'),
            'book_positions': book_stats.get('positions', 0),
            'game_stats': self.game_stats
        }
    