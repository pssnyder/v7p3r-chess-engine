# v7p3r_book.py

"""Opening Book for V7P3R Chess Engine
Handles basic opening moves from PGN files.
"""

import chess
import chess.pgn
import os
import random
import time

class OpeningBook:
    def __init__(self, opening_dir="pgn_opening_data"):
        self.opening_dir = opening_dir
        self.book_moves = {}
        self.max_book_moves = 10
        self.load_openings()
    
    def load_openings(self):
        """Load opening moves from PGN files"""
        print("Loading opening book...")
        start_time = time.time()
        
        if not os.path.exists(self.opening_dir):
            print(f"Opening directory {self.opening_dir} not found")
            return
        
        files_loaded = 0
        max_files = 10  # Limit number of files for faster startup
        
        for filename in os.listdir(self.opening_dir):
            if filename.endswith('.pgn') and files_loaded < max_files:
                filepath = os.path.join(self.opening_dir, filename)
                self._load_pgn_file(filepath)
                files_loaded += 1
                
                # Timeout after 10 seconds
                if time.time() - start_time > 10:
                    print("Opening book loading timeout - continuing with partial book")
                    break
        
        elapsed = time.time() - start_time
        print(f"Opening book loaded in {elapsed:.2f}s ({len(self.book_moves)} positions)")
    
    def _load_pgn_file(self, filepath):
        """Load moves from a single PGN file"""
        try:
            games_loaded = 0
            max_games_per_file = 5  # Limit games per file for faster loading
            
            with open(filepath, 'r') as f:
                while games_loaded < max_games_per_file:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    self._process_game(game)
                    games_loaded += 1
                    
            print(f"Loaded {games_loaded} games from {os.path.basename(filepath)}")
            
        except Exception as e:
            print(f"Error loading opening file {filepath}: {e}")
    
    def _process_game(self, game):
        """Process a single game and extract opening moves"""
        board = game.board()
        moves_played = 0
        
        for move in game.mainline_moves():
            if moves_played >= self.max_book_moves:
                break
            
            # Get current position as FEN (without move counters for transposition)
            position_key = self._get_position_key(board)
            
            # Add this move to the book for this position
            if position_key not in self.book_moves:
                self.book_moves[position_key] = []
            
            if move not in self.book_moves[position_key]:
                self.book_moves[position_key].append(move)
            
            # Make the move
            board.push(move)
            moves_played += 1
    
    def _get_position_key(self, board):
        """Get a position key for the book (FEN without move counters)"""
        fen = board.fen()
        # Remove halfmove and fullmove counters for transposition
        parts = fen.split()
        return ' '.join(parts[:4])
    
    def get_book_move(self, board):
        """Get a book move for the current position"""
        position_key = self._get_position_key(board)
        
        if position_key in self.book_moves:
            candidate_moves = self.book_moves[position_key]
            
            # Filter for legal moves (in case of position transpositions)
            legal_book_moves = []
            for move in candidate_moves:
                if move in board.legal_moves:
                    legal_book_moves.append(move)
            
            if legal_book_moves:
                # Return a random book move (or could be weighted)
                return random.choice(legal_book_moves)
        
        return None
    
    def is_in_book(self, board):
        """Check if current position is in the opening book"""
        position_key = self._get_position_key(board)
        return position_key in self.book_moves
    
    def get_book_statistics(self):
        """Get statistics about the loaded opening book"""
        total_positions = len(self.book_moves)
        total_moves = sum(len(moves) for moves in self.book_moves.values())
        
        return {
            'positions': total_positions,
            'total_moves': total_moves,
            'avg_moves_per_position': total_moves / total_positions if total_positions > 0 else 0
        }
