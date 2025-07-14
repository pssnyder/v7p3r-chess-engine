# v7p3r_stockfish.py

"""Stockfish Interface for V7P3R Chess Engine
Handles communication with Stockfish engine for opponent play.
"""

import chess
from stockfish import Stockfish

class StockfishHandler:
    def __init__(self, config):
        self.config = config
        stockfish_config = config.get_stockfish_config()
        
        # Stockfish configuration
        self.stockfish_path = stockfish_config.get('stockfish_path', 'stockfish.exe')
        self.elo_rating = stockfish_config.get('elo_rating', 400)
        self.depth = stockfish_config.get('depth', 4)
        self.debug_mode = stockfish_config.get('debug_mode', False)
        
        # Initialize Stockfish
        self.stockfish = None
        self._initialize_stockfish()
    
    def _initialize_stockfish(self):
        """Initialize Stockfish engine with configuration"""
        try:
            # Stockfish settings
            stockfish_params = {
                "Debug Log File": "",
                "Contempt": 0,
                "Min Split Depth": 0,
                "Threads": 1,
                "Ponder": "false",
                "Hash": 16,
                "MultiPV": 1,
                "Skill Level": 20,
                "Move Overhead": 10,
                "Minimum Thinking Time": 20,
                "Slow Mover": 100,
                "UCI_Chess960": "false",
            }
            
            self.stockfish = Stockfish(
                path=self.stockfish_path,
                depth=self.depth,
                parameters=stockfish_params
            )
            
            # Set ELO rating to limit strength
            if self.elo_rating < 2850:  # Only set if limiting strength
                self.stockfish.set_elo_rating(self.elo_rating)
            
            print(f"Stockfish initialized - ELO: {self.elo_rating}, Depth: {self.depth}")
            
        except Exception as e:
            print(f"Error initializing Stockfish: {e}")
            self.stockfish = None
    
    def is_available(self):
        """Check if Stockfish is available and ready"""
        return self.stockfish is not None and self.stockfish.is_fen_valid("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    def get_move(self, board):
        """Get Stockfish's move for the current position"""
        if not self.is_available():
            return None
        
        try:
            # Set the position
            fen = board.fen()
            self.stockfish.set_fen_position(fen)
            
            # Get the best move
            best_move_uci = self.stockfish.get_best_move()
            
            if best_move_uci:
                # Convert UCI move to chess.Move
                try:
                    move = chess.Move.from_uci(best_move_uci)
                    
                    # Validate move is legal
                    if move in board.legal_moves:
                        return move
                    else:
                        print(f"Stockfish suggested illegal move: {best_move_uci}")
                        return None
                        
                except ValueError:
                    print(f"Invalid UCI move from Stockfish: {best_move_uci}")
                    return None
            
            return None
            
        except Exception as e:
            print(f"Error getting Stockfish move: {e}")
            return None
    
    def get_evaluation(self, board):
        """Get Stockfish's evaluation of the position"""
        if not self.is_available():
            return None
        
        try:
            fen = board.fen()
            self.stockfish.set_fen_position(fen)
            
            evaluation = self.stockfish.get_evaluation()
            return evaluation
            
        except Exception as e:
            print(f"Error getting Stockfish evaluation: {e}")
            return None
    
    def get_top_moves(self, board, num_moves=3):
        """Get top moves from Stockfish"""
        if not self.is_available():
            return []
        
        try:
            fen = board.fen()
            self.stockfish.set_fen_position(fen)
            
            top_moves = self.stockfish.get_top_moves(num_moves)
            
            # Convert to chess.Move objects
            moves = []
            for move_info in top_moves:
                try:
                    move = chess.Move.from_uci(move_info['Move'])
                    if move in board.legal_moves:
                        moves.append({
                            'move': move,
                            'centipawn': move_info.get('Centipawn'),
                            'mate': move_info.get('Mate')
                        })
                except ValueError:
                    continue
            
            return moves
            
        except Exception as e:
            print(f"Error getting Stockfish top moves: {e}")
            return []
    
    def set_depth(self, depth):
        """Set Stockfish search depth"""
        self.depth = depth
        if self.stockfish:
            self.stockfish.set_depth(depth)
    
    def set_elo(self, elo):
        """Set Stockfish ELO rating"""
        self.elo_rating = elo
        if self.stockfish:
            self.stockfish.set_elo_rating(elo)
    
    def quit(self):
        """Clean up Stockfish resources"""
        if self.stockfish:
            try:
                self.stockfish = None
            except:
                pass
