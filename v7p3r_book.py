# v7p3r_engine/v7p3r_book.py
# Focused opening book for v7p3r chess engine
# Contains manually curated mainlines for fast access

import chess
import random

class v7p3rBook:
    """
    V7P3R Chess Engine Opening Book
    
    A focused, fast-access opening book containing manually curated mainlines
    for a specific repertoire designed for practical play.
    
    WHITE REPERTOIRE:
    - London System: 1.d4 with Bf4, e3, Nf3, Bd3, h3, Nbd2
    - Vienna Game: 1.e4 e5 2.Nc3 for sharp tactical play
    - Queen's Gambit: 1.d4 d5 2.c4 for positional control
    - Italian Game: 1.e4 e5 2.Nf3 Nc6 3.Bc4 for classical development
    - Ruy Lopez: 1.e4 e5 2.Nf3 Nc6 3.Bb5 for strategic complexity
    
    BLACK REPERTOIRE:
    - Caro-Kann: 1...c6 against 1.e4 for solid structure
    - French Defense: 1...e6 against 1.e4 for strategic counterplay
    - Scandinavian: 1...d5 against 1.e4 for immediate central activity
    - King's Indian: 1...Nf6 against 1.d4 for dynamic kingside attack
    - Dutch Defense: 1...f5 against 1.d4 for aggressive play
    
    Each line is capped at 12 moves maximum, with some resolving earlier
    when reaching characteristic middlegame positions.
    """
    
    def __init__(self):
        self.book = {}
        self.off_book = False  # Flag to indicate when engine is out of book
        self._populate_book()

    def _populate_book(self):
        """Populate the opening book with manually curated mainlines"""
        # Starting position
        start_fen = chess.Board().fen()
        
        # WHITE'S FIRST MOVES
        self.book[start_fen] = [
            (chess.Move.from_uci("d2d4"), 35),  # London System, Queen's Gambit
            (chess.Move.from_uci("e2e4"), 30),  # Vienna, Italian, Ruy Lopez
            (chess.Move.from_uci("g1f3"), 20),  # Flexible development
            (chess.Move.from_uci("c2c4"), 15),  # English Opening
        ]
        
        # ==========================================
        # WHITE REPERTOIRE - LONDON SYSTEM
        # ==========================================
        self._add_london_system()
        
        # ==========================================
        # WHITE REPERTOIRE - 1.e4 OPENINGS  
        # ==========================================
        self._add_vienna_game()
        self._add_italian_game()
        self._add_ruy_lopez()
        
        # ==========================================
        # WHITE REPERTOIRE - QUEEN'S GAMBIT
        # ==========================================
        self._add_queens_gambit()
        
        # ==========================================
        # BLACK REPERTOIRE vs 1.e4
        # ==========================================
        self._add_caro_kann()
        self._add_french_defense()
        self._add_scandinavian()
        
        # ==========================================
        # BLACK REPERTOIRE vs 1.d4
        # ==========================================
        self._add_kings_indian()
        self._add_dutch_defense()
        
        # Reset off_book flag after populating
        self.off_book = False

    def _add_london_system(self):
        """Add London System mainlines to the book"""
        board = chess.Board()
        
        # 1.d4 - various Black responses
        board.push(chess.Move.from_uci("d2d4"))
        fen_after_d4 = board.fen()
        self.book[fen_after_d4] = [
            (chess.Move.from_uci("d7d5"), 35),    # Most common
            (chess.Move.from_uci("g8f6"), 25),    # King's Indian/Nimzo setup
            (chess.Move.from_uci("f7f5"), 15),    # Dutch Defense
            (chess.Move.from_uci("e7e6"), 12),    # French/Queen's Indian
            (chess.Move.from_uci("c7c5"), 10),    # Benoni systems
            (chess.Move.from_uci("c7c6"), 3),     # Caro-Kann move order
        ]
        
        # London vs 1...d5 (most common mainline)
        board.push(chess.Move.from_uci("d7d5"))  # 1...d5
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("c1f4"), 50)]  # 2.Bf4
        
        board.push(chess.Move.from_uci("c1f4"))  # 2.Bf4
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("g8f6"), 40),    # Most natural
            (chess.Move.from_uci("c7c5"), 25),    # Aggressive try
            (chess.Move.from_uci("e7e6"), 20),    # Solid
            (chess.Move.from_uci("c8f5"), 15),    # Bishop development
        ]
        
        # London vs 1...d5 2.Bf4 Nf6
        board.push(chess.Move.from_uci("g8f6"))  # 2...Nf6
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("e2e3"), 50)]  # 3.e3
        
        board.push(chess.Move.from_uci("e2e3"))  # 3.e3
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("e7e6"), 30),    # French structure
            (chess.Move.from_uci("c7c5"), 25),    # Central challenge
            (chess.Move.from_uci("c8f5"), 20),    # Bishop development
            (chess.Move.from_uci("b8d7"), 15),    # Knight development
            (chess.Move.from_uci("g7g6"), 10),    # King's Indian approach
        ]
        
        # Continue main line: 3...e6
        board.push(chess.Move.from_uci("e7e6"))  # 3...e6
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("g1f3"), 50)]  # 4.Nf3
        
        board.push(chess.Move.from_uci("g1f3"))  # 4.Nf3
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("c7c5"), 30),    # Central break
            (chess.Move.from_uci("b8d7"), 25),    # Development
            (chess.Move.from_uci("f8d6"), 20),    # Bishop trade offer
            (chess.Move.from_uci("c8d7"), 15),    # Simple development
            (chess.Move.from_uci("b7b6"), 10),    # Queen's Indian style
        ]
        
        # Continue: 4...c5
        board.push(chess.Move.from_uci("c7c5"))  # 4...c5
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("c2c3"), 50)]  # 5.c3
        
        board.push(chess.Move.from_uci("c2c3"))  # 5.c3
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("b8c6"), 30),    # Development
            (chess.Move.from_uci("b8d7"), 25),    # Alternative knight
            (chess.Move.from_uci("f8d6"), 20),    # Bishop trade
            (chess.Move.from_uci("a7a6"), 15),    # Queen's side play
            (chess.Move.from_uci("d8c7"), 10),    # Centralization
        ]
        
        # Complete London setup: 5...Nc6 6.Bd3
        board.push(chess.Move.from_uci("b8c6"))  # 5...Nc6
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("f1d3"), 50)]  # 6.Bd3
        
        board.push(chess.Move.from_uci("f1d3"))  # 6.Bd3
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("f8d6"), 35),    # Natural development
            (chess.Move.from_uci("c5d4"), 25),    # Central break
            (chess.Move.from_uci("f8e7"), 20),    # Solid
            (chess.Move.from_uci("c8d7"), 15),    # Development
            (chess.Move.from_uci("a7a6"), 5),     # Slow
        ]
        
        # Final London position: 6...Bd6
        board.push(chess.Move.from_uci("f8d6"))  # 6...Bd6
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("d6f4"), 50)]  # 7.Bxf4 (bishop trade)
        
        # This completes the main London line at move 7/8 - good stopping point

    def _add_vienna_game(self):
        """Add Vienna Game mainlines to the book"""
        board = chess.Board()
        
        # 1.e4 e5
        board.push(chess.Move.from_uci("e2e4"))
        board.push(chess.Move.from_uci("e7e5"))
        fen = board.fen()
        # Add Vienna as option after 1.e4 e5
        if fen in self.book:
            self.book[fen].append((chess.Move.from_uci("b1c3"), 35))
        else:
            self.book[fen] = [
                (chess.Move.from_uci("g1f3"), 40),    # Italian/Ruy Lopez
                (chess.Move.from_uci("b1c3"), 35),    # Vienna Game
                (chess.Move.from_uci("f1c4"), 15),    # Bishop's Opening
                (chess.Move.from_uci("d2d3"), 10),    # King's Indian Attack
            ]
        
        # 2.Nc3
        board.push(chess.Move.from_uci("b1c3"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("g8f6"), 35),    # Most popular
            (chess.Move.from_uci("b8c6"), 30),    # Classical
            (chess.Move.from_uci("f8c5"), 20),    # Bishop development
            (chess.Move.from_uci("f7f5"), 15),    # Aggressive Vienna Gambit
        ]
        
        # Vienna vs 2...Nf6 (most common)
        board.push(chess.Move.from_uci("g8f6"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("f2f4"), 50)]  # 3.f4 Vienna Game proper
        
        board.push(chess.Move.from_uci("f2f4"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("d7d5"), 30),    # Counter in center
            (chess.Move.from_uci("b8c6"), 25),    # Development
            (chess.Move.from_uci("e5f4"), 20),    # Capture pawn
            (chess.Move.from_uci("f8b4"), 15),    # Pin knight
            (chess.Move.from_uci("d7d6"), 10),    # Solid
        ]
        
        # Continue with 3...d5
        board.push(chess.Move.from_uci("d7d5"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("f4e5"), 50)]  # 4.fxe5
        
        board.push(chess.Move.from_uci("f4e5"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("f6e4"), 40),    # Active knight
            (chess.Move.from_uci("f6d7"), 30),    # Retreat
            (chess.Move.from_uci("f6g4"), 20),    # Aggressive
            (chess.Move.from_uci("d5e4"), 10),    # Recapture
        ]
        
        # 4...Nxe4 main line
        board.push(chess.Move.from_uci("f6e4"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("g1f3"), 50)]  # 5.Nf3
        
        # This gives us a good Vienna Game setup by move 5

    def _add_italian_game(self):
        """Add Italian Game mainlines to the book"""
        board = chess.Board()
        
        # 1.e4 e5 2.Nf3
        board.push(chess.Move.from_uci("e2e4"))
        board.push(chess.Move.from_uci("e7e5"))
        board.push(chess.Move.from_uci("g1f3"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("b8c6"), 45),    # Most common
            (chess.Move.from_uci("d7d6"), 20),    # Philidor
            (chess.Move.from_uci("f7f5"), 15),    # Latvian Gambit
            (chess.Move.from_uci("g8f6"), 10),    # Petrov
            (chess.Move.from_uci("f8e7"), 5),     # Hungarian Defense
            (chess.Move.from_uci("d7d5"), 5),     # Scandinavian move order
        ]
        
        # 2...Nc6 3.Bc4 (Italian Game)
        board.push(chess.Move.from_uci("b8c6"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("f1c4"), 45),    # Italian Game
            (chess.Move.from_uci("f1b5"), 35),    # Ruy Lopez
            (chess.Move.from_uci("d2d4"), 15),    # Scotch Game
            (chess.Move.from_uci("c2c3"), 5),     # Ponziani
        ]
        
        board.push(chess.Move.from_uci("f1c4"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("f8c5"), 35),    # Italian Game main line
            (chess.Move.from_uci("f8e7"), 25),    # Hungarian Defense
            (chess.Move.from_uci("g8f6"), 20),    # Two Knights Defense
            (chess.Move.from_uci("f7f5"), 15),    # Rousseau Gambit
            (chess.Move.from_uci("d7d6"), 5),     # Paris Defense
        ]
        
        # Italian Game proper: 3...Bc5
        board.push(chess.Move.from_uci("f8c5"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("c2c3"), 40),    # Main line
            (chess.Move.from_uci("d2d3"), 30),    # Giuoco Pianissimo
            (chess.Move.from_uci("b2b4"), 20),    # Evans Gambit
            (chess.Move.from_uci("e1g1"), 10),    # Quick castle
        ]
        
        # 4.c3 main line
        board.push(chess.Move.from_uci("c2c3"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("g8f6"), 35),    # Most natural
            (chess.Move.from_uci("d7d6"), 25),    # Solid
            (chess.Move.from_uci("f7f5"), 20),    # Aggressive
            (chess.Move.from_uci("b8e7"), 15),    # Retreat
            (chess.Move.from_uci("a7a6"), 5),     # Slow
        ]
        
        # 4...Nf6 5.d4
        board.push(chess.Move.from_uci("g8f6"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("d2d4"), 50)]
        
        board.push(chess.Move.from_uci("d2d4"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("e5d4"), 40),    # Capture
            (chess.Move.from_uci("c5d6"), 30),    # Retreat
            (chess.Move.from_uci("c5b6"), 20),    # Retreat
            (chess.Move.from_uci("f6e4"), 10),    # Knight to center
        ]
        
        # This gives us solid Italian Game setup

    def _add_ruy_lopez(self):
        """Add Ruy Lopez mainlines to the book"""
        board = chess.Board()
        
        # Navigate to 1.e4 e5 2.Nf3 Nc6 position
        board.push(chess.Move.from_uci("e2e4"))
        board.push(chess.Move.from_uci("e7e5"))
        board.push(chess.Move.from_uci("g1f3"))
        board.push(chess.Move.from_uci("b8c6"))
        
        # This position should already exist from Italian, just add Ruy Lopez
        fen = board.fen()
        # The book entry should already exist from Italian Game, so we ensure Bb5 is included
        
        # 3.Bb5 Ruy Lopez
        board.push(chess.Move.from_uci("f1b5"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("a7a6"), 40),    # Morphy Defense
            (chess.Move.from_uci("g8f6"), 25),    # Berlin Defense
            (chess.Move.from_uci("f7f5"), 15),    # Schliemann
            (chess.Move.from_uci("f8c5"), 10),    # Classical Defense
            (chess.Move.from_uci("d7d6"), 5),     # Steinitz Defense
            (chess.Move.from_uci("g7g6"), 5),     # Smyslov Defense
        ]
        
        # Morphy Defense: 3...a6
        board.push(chess.Move.from_uci("a7a6"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("b5a4"), 45),    # Main line
            (chess.Move.from_uci("b5c6"), 35),    # Exchange Variation
            (chess.Move.from_uci("b5c4"), 20),    # Italian transpose
        ]
        
        # 4.Ba4 main line
        board.push(chess.Move.from_uci("b5a4"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("g8f6"), 40),    # Most common
            (chess.Move.from_uci("f7f5"), 25),    # Schliemann delayed
            (chess.Move.from_uci("d7d6"), 20),    # Steinitz deferred
            (chess.Move.from_uci("b7b5"), 10),    # Aggressive
            (chess.Move.from_uci("f8c5"), 5),     # Modern Steinitz
        ]
        
        # 4...Nf6 5.O-O
        board.push(chess.Move.from_uci("g8f6"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("e1g1"), 50)]
        
        board.push(chess.Move.from_uci("e1g1"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("f8e7"), 35),    # Closed Ruy Lopez
            (chess.Move.from_uci("f6e4"), 25),    # Open Defense
            (chess.Move.from_uci("b7b5"), 20),    # Norwegian Defense
            (chess.Move.from_uci("f8c5"), 15),    # Modern Defense
            (chess.Move.from_uci("d7d6"), 5),     # Old Defense
        ]
        
        # Closed Ruy Lopez: 5...Be7
        board.push(chess.Move.from_uci("f8e7"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("f1e1"), 50)]  # 6.Re1
        
        board.push(chess.Move.from_uci("f1e1"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("b7b5"), 35),    # Most common
            (chess.Move.from_uci("d7d6"), 30),    # Solid
            (chess.Move.from_uci("e8g8"), 25),    # Castle first
            (chess.Move.from_uci("f6d7"), 10),    # Cozio Defense
        ]
        
        # This gives us a solid Ruy Lopez opening

    def _add_queens_gambit(self):
        """Add Queen's Gambit mainlines to the book"""
        board = chess.Board()
        
        # 1.d4 d5
        board.push(chess.Move.from_uci("d2d4"))
        board.push(chess.Move.from_uci("d7d5"))
        fen = board.fen()
        if fen in self.book:
            self.book[fen].append((chess.Move.from_uci("c2c4"), 45))
        else:
            self.book[fen] = [
                (chess.Move.from_uci("c2c4"), 45),    # Queen's Gambit
                (chess.Move.from_uci("c1f4"), 50),    # London System (from earlier)
            ]
        
        # 2.c4 Queen's Gambit
        board.push(chess.Move.from_uci("c2c4"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("d5c4"), 35),    # Queen's Gambit Accepted
            (chess.Move.from_uci("e7e6"), 30),    # Queen's Gambit Declined
            (chess.Move.from_uci("c7c6"), 20),    # Slav Defense
            (chess.Move.from_uci("g8f6"), 15),    # Nimzo move order
        ]
        
        # QGA: 2...dxc4
        board.push(chess.Move.from_uci("d5c4"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("g1f3"), 50)]  # 3.Nf3
        
        board.push(chess.Move.from_uci("g1f3"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("g8f6"), 35),    # Most natural
            (chess.Move.from_uci("a7a6"), 25),    # Hold the pawn
            (chess.Move.from_uci("c7c5"), 20),    # Central break
            (chess.Move.from_uci("e7e6"), 15),    # Solid development
            (chess.Move.from_uci("b8c6"), 5),     # Development
        ]
        
        # QGA main: 3...Nf6
        board.push(chess.Move.from_uci("g8f6"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("e2e3"), 50)]  # 4.e3
        
        board.push(chess.Move.from_uci("e2e3"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("e7e6"), 30),    # Solid
            (chess.Move.from_uci("c8g4"), 25),    # Pin
            (chess.Move.from_uci("c7c5"), 20),    # Central
            (chess.Move.from_uci("a7a6"), 15),    # Hold pawn
            (chess.Move.from_uci("b8c6"), 10),    # Development
        ]
        
        # 4...e6 5.Bxc4
        board.push(chess.Move.from_uci("e7e6"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("f1c4"), 50)]
        
        board.push(chess.Move.from_uci("f1c4"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("c7c5"), 35),    # Central break
            (chess.Move.from_uci("a7a6"), 25),    # Prepare b5
            (chess.Move.from_uci("b8c6"), 20),    # Development
            (chess.Move.from_uci("b7b5"), 15),    # Immediate expansion
            (chess.Move.from_uci("f8e7"), 5),     # Simple development
        ]
        
        # This gives us a solid Queen's Gambit structure

    def _add_caro_kann(self):
        """Add Caro-Kann Defense mainlines to the book"""
        board = chess.Board()
        
        # 1.e4 c6 (Caro-Kann)
        board.push(chess.Move.from_uci("e2e4"))
        fen = board.fen()
        # This position should exist from Vienna setup, add Caro-Kann
        if fen in self.book:
            self.book[fen].append((chess.Move.from_uci("c7c6"), 25))
        else:
            self.book[fen] = [
                (chess.Move.from_uci("e7e5"), 35),    # 1...e5
                (chess.Move.from_uci("c7c5"), 20),    # Sicilian
                (chess.Move.from_uci("c7c6"), 25),    # Caro-Kann
                (chess.Move.from_uci("e7e6"), 15),    # French
                (chess.Move.from_uci("d7d5"), 5),     # Scandinavian
            ]
        
        board.push(chess.Move.from_uci("c7c6"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("d2d4"), 45),    # Main line
            (chess.Move.from_uci("b1c3"), 25),    # Two Knights
            (chess.Move.from_uci("g1f3"), 20),    # King's Indian Attack
            (chess.Move.from_uci("f2f4"), 10),    # Fantasy Variation
        ]
        
        # 2.d4 d5 (main Caro-Kann)
        board.push(chess.Move.from_uci("d2d4"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("d7d5"), 50)]
        
        board.push(chess.Move.from_uci("d7d5"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("b1c3"), 35),    # Classical/Advance
            (chess.Move.from_uci("e4d5"), 30),    # Exchange Variation
            (chess.Move.from_uci("e4e5"), 20),    # Advance Variation
            (chess.Move.from_uci("g1f3"), 15),    # Two Knights
        ]
        
        # Main line: 3.Nc3 (or 3.Nd2)
        board.push(chess.Move.from_uci("b1c3"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("d5e4"), 40),    # Main line
            (chess.Move.from_uci("g8f6"), 25),    # Classical Defense
            (chess.Move.from_uci("g7g6"), 20),    # Modern Defense
            (chess.Move.from_uci("e7e6"), 15),    # Tal Variation
        ]
        
        # 3...dxe4 4.Nxe4
        board.push(chess.Move.from_uci("d5e4"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("c3e4"), 50)]
        
        board.push(chess.Move.from_uci("c3e4"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("c8f5"), 35),    # Main line
            (chess.Move.from_uci("g8f6"), 30),    # Bronstein-Larsen
            (chess.Move.from_uci("b8d7"), 20),    # Smyslov Variation
            (chess.Move.from_uci("h7h6"), 15),    # Keres Variation
        ]
        
        # 4...Bf5 main line
        board.push(chess.Move.from_uci("c8f5"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("e4g3"), 50)]  # 5.Ng3
        
        board.push(chess.Move.from_uci("e4g3"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("f5g6"), 40),    # Most common
            (chess.Move.from_uci("f5h7"), 30),    # Retreat
            (chess.Move.from_uci("f5e6"), 20),    # Centralization
            (chess.Move.from_uci("f5d7"), 10),    # Retreat
        ]
        
        # This gives us solid Caro-Kann structure

    def _add_french_defense(self):
        """Add French Defense mainlines to the book"""
        board = chess.Board()
        
        # 1.e4 e6 (French Defense)
        board.push(chess.Move.from_uci("e2e4"))
        fen = board.fen()
        # Add French to existing 1.e4 responses
        
        board.push(chess.Move.from_uci("e7e6"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("d2d4"), 45),    # Main line
            (chess.Move.from_uci("d2d3"), 25),    # King's Indian Attack
            (chess.Move.from_uci("g1f3"), 20),    # Réti System
            (chess.Move.from_uci("b1c3"), 10),    # Two Knights
        ]
        
        # 2.d4 d5 (French main)
        board.push(chess.Move.from_uci("d2d4"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("d7d5"), 50)]
        
        board.push(chess.Move.from_uci("d7d5"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("b1c3"), 35),    # Classical/Winawer
            (chess.Move.from_uci("e4d5"), 30),    # Exchange Variation
            (chess.Move.from_uci("e4e5"), 25),    # Advance Variation
            (chess.Move.from_uci("b1d2"), 10),    # Tarrasch Variation
        ]
        
        # Winawer/Classical: 3.Nc3
        board.push(chess.Move.from_uci("b1c3"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("f8b4"), 35),    # Winawer Variation
            (chess.Move.from_uci("g8f6"), 30),    # Classical Defense
            (chess.Move.from_uci("d5e4"), 20),    # Rubinstein Variation
            (chess.Move.from_uci("c7c5"), 15),    # Accelerated Dragon style
        ]
        
        # Winawer: 3...Bb4
        board.push(chess.Move.from_uci("f8b4"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("e4e5"), 40),    # Main Winawer
            (chess.Move.from_uci("c1d2"), 30),    # Positional
            (chess.Move.from_uci("a2a3"), 20),    # Immediate question
            (chess.Move.from_uci("g1e2"), 10),    # Unusual
        ]
        
        # 4.e5 (Winawer main)
        board.push(chess.Move.from_uci("e4e5"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("c7c5"), 40),    # Most common
            (chess.Move.from_uci("g8e7"), 25),    # Solid
            (chess.Move.from_uci("b8d7"), 20),    # Development
            (chess.Move.from_uci("h7h6"), 15),    # Prevent Ng5
        ]
        
        # 4...c5 5.a3
        board.push(chess.Move.from_uci("c7c5"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("a2a3"), 50)]
        
        board.push(chess.Move.from_uci("a2a3"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("b4c3"), 40),    # Main line
            (chess.Move.from_uci("b4a5"), 30),    # Retreat
            (chess.Move.from_uci("c5d4"), 20),    # Central break
            (chess.Move.from_uci("b4e7"), 10),    # Retreat
        ]
        
        # This gives us solid French structure

    def _add_scandinavian(self):
        """Add Scandinavian Defense mainlines to the book"""
        board = chess.Board()
        
        # 1.e4 d5 (Scandinavian)
        board.push(chess.Move.from_uci("e2e4"))
        fen = board.fen()
        # Add Scandinavian to existing responses (should already exist)
        
        board.push(chess.Move.from_uci("d7d5"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("e4d5"), 50)]  # 2.exd5
        
        board.push(chess.Move.from_uci("e4d5"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("d8d5"), 40),    # Modern main line
            (chess.Move.from_uci("g8f6"), 35),    # Marshall Gambit
            (chess.Move.from_uci("c7c6"), 20),    # Gubinsky-Melts Defense
            (chess.Move.from_uci("e7e6"), 5),     # Icelandic Gambit
        ]
        
        # Modern Scandinavian: 2...Qxd5
        board.push(chess.Move.from_uci("d8d5"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("b1c3"), 50)]  # 3.Nc3
        
        board.push(chess.Move.from_uci("b1c3"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("d5a5"), 35),    # Main line retreat
            (chess.Move.from_uci("d5d6"), 30),    # Central queen
            (chess.Move.from_uci("d5d8"), 20),    # Back home
            (chess.Move.from_uci("d5e6"), 15),    # Aggressive
        ]
        
        # 3...Qa5 main line
        board.push(chess.Move.from_uci("d5a5"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("d2d4"), 40),    # Center control
            (chess.Move.from_uci("g1f3"), 30),    # Development
            (chess.Move.from_uci("c1d2"), 20),    # Bishop development
            (chess.Move.from_uci("b2b4"), 10),    # Wing attack
        ]
        
        # 4.d4
        board.push(chess.Move.from_uci("d2d4"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("g8f6"), 35),    # Development
            (chess.Move.from_uci("c7c6"), 25),    # Prepare Nd7
            (chess.Move.from_uci("c8f5"), 20),    # Bishop development
            (chess.Move.from_uci("e7e6"), 15),    # Solid
            (chess.Move.from_uci("g7g6"), 5),     # Fianchetto
        ]
        
        # 4...Nf6 5.Nf3
        board.push(chess.Move.from_uci("g8f6"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("g1f3"), 50)]
        
        board.push(chess.Move.from_uci("g1f3"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("c8f5"), 35),    # Bishop development
            (chess.Move.from_uci("c7c6"), 30),    # Solid
            (chess.Move.from_uci("g7g6"), 20),    # Fianchetto
            (chess.Move.from_uci("e7e6"), 15),    # French structure
        ]
        
        # This gives us solid Scandinavian structure

    def _add_kings_indian(self):
        """Add King's Indian Defense mainlines to the book"""
        board = chess.Board()
        
        # 1.d4 Nf6 (King's Indian move order)
        board.push(chess.Move.from_uci("d2d4"))
        fen = board.fen()
        # Should already exist from London setup
        
        board.push(chess.Move.from_uci("g8f6"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("c2c4"), 40),    # King's Indian proper
            (chess.Move.from_uci("c1f4"), 30),    # London vs Nf6
            (chess.Move.from_uci("g1f3"), 20),    # Réti System
            (chess.Move.from_uci("b1c3"), 10),    # Veresov Attack
        ]
        
        # 2.c4 g6 (King's Indian setup)
        board.push(chess.Move.from_uci("c2c4"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("g7g6"), 35),    # King's Indian proper
            (chess.Move.from_uci("e7e6"), 25),    # Nimzo-Indian
            (chess.Move.from_uci("c7c5"), 20),    # Benoni systems
            (chess.Move.from_uci("d7d5"), 15),    # Queen's Gambit Declined
            (chess.Move.from_uci("e7e5"), 5),     # Budapest Gambit
        ]
        
        board.push(chess.Move.from_uci("g7g6"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("b1c3"), 50)]  # 3.Nc3
        
        board.push(chess.Move.from_uci("b1c3"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("f8g7"), 50)]  # 3...Bg7
        
        board.push(chess.Move.from_uci("f8g7"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("e2e4"), 50)]  # 4.e4
        
        board.push(chess.Move.from_uci("e2e4"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("d7d6"), 40),    # Traditional KID
            (chess.Move.from_uci("e8g8"), 25),    # Castle first
            (chess.Move.from_uci("c7c5"), 20),    # Benoni transpose
            (chess.Move.from_uci("d7d5"), 15),    # Petrosian System counter
        ]
        
        # 4...d6 5.f3 (Sämisch)
        board.push(chess.Move.from_uci("d7d6"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("f2f3"), 35),    # Sämisch Variation
            (chess.Move.from_uci("g1f3"), 30),    # Classical
            (chess.Move.from_uci("f1e2"), 20),    # Positional
            (chess.Move.from_uci("h2h3"), 15),    # Makogonov
        ]
        
        # Sämisch: 5.f3 O-O
        board.push(chess.Move.from_uci("f2f3"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("e8g8"), 50)]
        
        board.push(chess.Move.from_uci("e8g8"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("c1e3"), 50)]  # 6.Be3
        
        board.push(chess.Move.from_uci("c1e3"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("e7e5"), 40),    # Main KID break
            (chess.Move.from_uci("c7c5"), 25),    # Benoni style
            (chess.Move.from_uci("b8d7"), 20),    # Development
            (chess.Move.from_uci("a7a6"), 15),    # Prepare b5
        ]
        
        # This gives us classic King's Indian structure

    def _add_dutch_defense(self):
        """Add Dutch Defense mainlines to the book"""
        board = chess.Board()
        
        # 1.d4 f5 (Dutch Defense)
        board.push(chess.Move.from_uci("d2d4"))
        fen = board.fen()
        # Should already exist from London, add Dutch
        
        board.push(chess.Move.from_uci("f7f5"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("g1f3"), 35),    # Most common
            (chess.Move.from_uci("c2c4"), 30),    # Active center
            (chess.Move.from_uci("b1c3"), 20),    # Development
            (chess.Move.from_uci("c1f4"), 15),    # London vs Dutch
        ]
        
        # 2.Nf3 Nf6 (classical Dutch setup)
        board.push(chess.Move.from_uci("g1f3"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("g8f6"), 40),    # Most natural
            (chess.Move.from_uci("e7e6"), 25),    # French structure
            (chess.Move.from_uci("d7d6"), 20),    # Solid
            (chess.Move.from_uci("g7g6"), 15),    # Leningrad Dutch
        ]
        
        board.push(chess.Move.from_uci("g8f6"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("g2g3"), 50)]  # 3.g3 (fianchetto setup)
        
        board.push(chess.Move.from_uci("g2g3"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("e7e6"), 35),    # Classical Dutch
            (chess.Move.from_uci("g7g6"), 30),    # Leningrad Dutch
            (chess.Move.from_uci("d7d6"), 20),    # Solid
            (chess.Move.from_uci("c7c6"), 15),    # Prepare Qc7
        ]
        
        # Classical Dutch: 3...e6
        board.push(chess.Move.from_uci("e7e6"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("f1g2"), 50)]  # 4.Bg2
        
        board.push(chess.Move.from_uci("f1g2"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("f8e7"), 35),    # Standard development
            (chess.Move.from_uci("d7d5"), 25),    # Stonewall setup
            (chess.Move.from_uci("f8d6"), 20),    # Bishop development
            (chess.Move.from_uci("b8d7"), 15),    # Knight development
            (chess.Move.from_uci("c7c6"), 5),     # Solid
        ]
        
        # 4...Be7 5.O-O
        board.push(chess.Move.from_uci("f8e7"))
        fen = board.fen()
        self.book[fen] = [(chess.Move.from_uci("e1g1"), 50)]
        
        board.push(chess.Move.from_uci("e1g1"))
        fen = board.fen()
        self.book[fen] = [
            (chess.Move.from_uci("e8g8"), 40),    # Castle
            (chess.Move.from_uci("d7d6"), 25),    # Solid setup
            (chess.Move.from_uci("d7d5"), 20),    # Stonewall
            (chess.Move.from_uci("b8d7"), 15),    # Development
        ]
        
        # This gives us solid Dutch Defense structure

    def get_book_move(self, board):
        """
        Get a move from the opening book if position is in the book.
        Sets off_book flag to True if no book move is found.
        
        Returns:
            chess.Move or None: Book move if found, None otherwise
        """
        fen = board.fen()
        if fen in self.book:
            moves = self.book[fen]
            if moves:  # Ensure there are moves available
                # Choose a move based on weights
                total_weight = sum(weight for _, weight in moves)
                if total_weight > 0:
                    choice = random.randint(1, total_weight)
                    current_weight = 0
                    for move, weight in moves:
                        current_weight += weight
                        if choice <= current_weight:
                            return move

        # No book move found - set off_book flag
        self.off_book = True
        return None

    def is_off_book(self):
        """
        Check if the engine is currently out of book.
        
        Returns:
            bool: True if engine is out of book, False otherwise
        """
        return self.off_book

    def reset_book_status(self):
        """Reset the off_book flag (useful for new games)"""
        self.off_book = False

    def has_book_move(self, board):
        """
        Check if the current position has any book moves without consuming one.
        
        Args:
            board (chess.Board): Current position
            
        Returns:
            bool: True if position is in book with available moves
        """
        fen = board.fen()
        return fen in self.book and len(self.book[fen]) > 0

    def get_book_moves(self, board):
        """
        Get all available book moves for a position with their weights.
        
        Args:
            board (chess.Board): Current position
            
        Returns:
            list: List of (move, weight) tuples, empty if not in book
        """
        fen = board.fen()
        if fen in self.book:
            return self.book[fen].copy()
        return []

    def add_position(self, board, move, weight=10):
        """
        Add a position-move pair to the opening book.
        
        Args:
            board (chess.Board): Position to add
            move (chess.Move): Move to add for this position
            weight (int): Weight for this move (higher = more likely to be chosen)
        """
        fen = board.fen()
        if fen not in self.book:
            self.book[fen] = []

        # Check if move already exists
        for i, (existing_move, existing_weight) in enumerate(self.book[fen]):
            if existing_move == move:
                # Update weight
                self.book[fen][i] = (move, existing_weight + weight)
                return

        # Add new move
        self.book[fen].append((move, weight))

    def get_book_size(self):
        """
        Get the number of positions in the opening book.
        
        Returns:
            int: Number of positions with book moves
        """
        return len(self.book)

    def get_book_statistics(self):
        """
        Get statistics about the opening book.
        
        Returns:
            dict: Dictionary with book statistics
        """
        total_positions = len(self.book)
        total_moves = sum(len(moves) for moves in self.book.values())
        avg_moves_per_position = total_moves / total_positions if total_positions > 0 else 0
        
        return {
            'total_positions': total_positions,
            'total_moves': total_moves,
            'avg_moves_per_position': round(avg_moves_per_position, 2),
            'off_book': self.off_book
        }

    def save_to_file(self, filename='v7p3r_opening_book.txt'):
        """
        Save the opening book to a file.
        
        Args:
            filename (str): Filename to save to
        """
        with open(filename, 'w') as f:
            f.write("# V7P3R Opening Book\n")
            f.write("# Format: FEN|MOVE_UCI|WEIGHT\n")
            for fen, moves in self.book.items():
                for move, weight in moves:
                    f.write(f"{fen}|{move.uci()}|{weight}\n")

    def load_from_file(self, filename='v7p3r_opening_book.txt'):
        """
        Load the opening book from a file.
        
        Args:
            filename (str): Filename to load from
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        self.book = {}
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            fen, move_uci, weight = line.split('|')
                            if fen not in self.book:
                                self.book[fen] = []
                            self.book[fen].append((chess.Move.from_uci(move_uci), int(weight)))
                        except (ValueError, chess.InvalidMoveError):
                            # Skip invalid lines
                            continue
            self.off_book = False
            return True
        except FileNotFoundError:
            print(f"Opening book file {filename} not found.")
            return False


# Simple opening principles class for evaluation when out of book
class OpeningPrinciples:
    """
    Provides evaluation methods for opening principles when the engine
    is out of book. This helps guide play in unfamiliar opening positions.
    """
    
    @staticmethod
    def evaluate_opening_principles(board):
        """
        Evaluate adherence to opening principles.
        
        Args:
            board (chess.Board): Position to evaluate
            
        Returns:
            int: Score based on opening principles (positive = good for current player)
        """
        score = 0
        move_count = len(board.move_stack) // 2  # Full moves

        # Only apply in the first 12 moves
        if move_count > 12:
            return 0

        # 1. Control the center with pawns or pieces
        score += OpeningPrinciples._evaluate_center_control(board)

        # 2. Develop knights and bishops
        score += OpeningPrinciples._evaluate_piece_development(board)

        # 3. Castle early
        score += OpeningPrinciples._evaluate_castling(board)

        # 4. Don't move same piece twice unnecessarily
        score += OpeningPrinciples._evaluate_piece_efficiency(board)

        # 5. Don't bring queen out early
        score += OpeningPrinciples._evaluate_queen_development(board)

        # Return score from perspective of current player
        return score if board.turn == chess.WHITE else -score

    @staticmethod
    def _evaluate_center_control(board):
        """Evaluate center control"""
        score = 0
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]

        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                if piece.piece_type == chess.PAWN:
                    score += 15 if piece.color == chess.WHITE else -15
                else:
                    score += 10 if piece.color == chess.WHITE else -10

            # Attacks on center squares
            if board.is_attacked_by(chess.WHITE, square):
                score += 5
            if board.is_attacked_by(chess.BLACK, square):
                score -= 5

        return score

    @staticmethod
    def _evaluate_piece_development(board):
        """Evaluate development of knights and bishops"""
        score = 0

        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            
            # Knight development
            knight_squares = [chess.B1, chess.G1] if color == chess.WHITE else [chess.B8, chess.G8]
            for square in knight_squares:
                piece = board.piece_at(square)
                if not piece or piece.piece_type != chess.KNIGHT or piece.color != color:
                    score += 10 * sign

            # Bishop development
            bishop_squares = [chess.C1, chess.F1] if color == chess.WHITE else [chess.C8, chess.F8]
            for square in bishop_squares:
                piece = board.piece_at(square)
                if not piece or piece.piece_type != chess.BISHOP or piece.color != color:
                    score += 10 * sign

        return score

    @staticmethod
    def _evaluate_castling(board):
        """Evaluate castling"""
        score = 0

        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            king_square = board.king(color)

            # Bonus for having castled
            if ((color == chess.WHITE and king_square in [chess.G1, chess.C1]) or 
                (color == chess.BLACK and king_square in [chess.G8, chess.C8])):
                score += 30 * sign

            # Penalty for losing castling rights without castling
            elif not board.has_castling_rights(color):
                if ((color == chess.WHITE and king_square == chess.E1) or 
                    (color == chess.BLACK and king_square == chess.E8)):
                    score -= 20 * sign

        return score

    @staticmethod
    def _evaluate_piece_efficiency(board):
        """Evaluate piece placement and efficiency"""
        score = 0

        # Knights are better in center
        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            for square in board.pieces(chess.KNIGHT, color):
                # Calculate distance from center
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                file_distance = abs(file - 3.5)
                rank_distance = abs(rank - 3.5)
                distance = file_distance + rank_distance
                score += (4 - distance) * 2 * sign

        # Bishops on long diagonals
        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            for square in board.pieces(chess.BISHOP, color):
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                # Check if on major diagonal
                if abs(file - rank) == 0 or abs(file - (7 - rank)) == 0:
                    score += 5 * sign

        return score

    @staticmethod
    def _evaluate_queen_development(board):
        """Evaluate queen development timing"""
        score = 0
        move_count = len(board.move_stack) // 2

        # Penalty for early queen development (first 6 moves)
        if move_count < 6:
            for color in [chess.WHITE, chess.BLACK]:
                sign = 1 if color == chess.WHITE else -1
                start_square = chess.D1 if color == chess.WHITE else chess.D8
                
                queen_square = None
                for square in board.pieces(chess.QUEEN, color):
                    queen_square = square
                    break
                
                if queen_square and queen_square != start_square:
                    score -= 15 * sign

        return score
