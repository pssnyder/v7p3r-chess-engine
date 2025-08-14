import unittest
import chess
import os
import sys
import time
from typing import Optional

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_search import v7p3rSearch
from v7p3r_rules import v7p3rRules
from v7p3r_score import v7p3rScore
from v7p3r_pst import v7p3rPST
from v7p3r_time import v7p3rTime
from v7p3r_book import v7p3rBook
from v7p3r_config import v7p3rConfig
from v7p3r_ordering import v7p3rOrdering

class TestShortCircuit(unittest.TestCase):
    def setUp(self):
        self.config = v7p3rConfig()
        self.rules = v7p3rRules()
        self.pst = v7p3rPST()
        self.score = v7p3rScore(self.rules, self.pst)
        self.time = v7p3rTime()
        self.book = v7p3rBook()
        self.ordering = v7p3rOrdering(scoring_calculator=self.score)
        
        self.search = v7p3rSearch(
            scoring_calculator=self.score,
            move_organizer=self.ordering,
            time_manager=self.time,
            opening_book=self.book,
            rules_manager=self.rules
        )
        
    def test_checkmate_in_one(self):
        """Test immediate checkmate detection."""
        # Position with mate in 1
        board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
        
        # Test direct mate finder
        mate_move = self.search.find_checkmate_in_n(board, 1)
        self.assertNotEqual(mate_move, chess.Move.null())
        board.push(mate_move)
        self.assertTrue(board.is_checkmate())
        
        # Test via iterative deepening
        board.pop()
        id_move = self.search.iterative_deepening_search(board, board.turn)
        self.assertEqual(mate_move, id_move)
        
    def test_checkmate_in_two(self):
        """Test detection of checkmate in two moves."""
        # Position leading to mate in two (famous puzzle position)
        board = chess.Board("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1")
        
        # Test via direct mate finder
        mate_move = self.search.find_checkmate_in_n(board, 2)
        self.assertNotEqual(mate_move, chess.Move.null())
        
        # Test that it's not an immediate checkmate but leads to a forced mate
        board.push(mate_move)
        self.assertFalse(board.is_checkmate())
        
        # Find best defense
        best_defense: Optional[chess.Move] = None
        best_defense_score = float('inf')
        
        for defense in board.legal_moves:
            test_board = board.copy()
            test_board.push(defense)
            if test_board.is_checkmate():
                continue
                
            score = -self.score.evaluate_position(test_board)
            if score < best_defense_score:
                best_defense_score = score
                best_defense = defense
                
        self.assertIsNotNone(best_defense, "Should have found a defense move")
        if best_defense:
            board.push(best_defense)
            
            # Now we should find mate in 1
            mate_in_one = self.search.find_checkmate_in_n(board, 1)
            self.assertNotEqual(mate_in_one, chess.Move.null())
            
            board.push(mate_in_one)
            self.assertTrue(board.is_checkmate())
            
            # Clean up
            board.pop()  # mate in one
            board.pop()  # defense
        
        board.pop()  # mate move
        
    def test_stalemate_prevention(self):
        """Test that the engine avoids stalemate when better alternatives exist."""
        # Position where queen could give stalemate
        board = chess.Board("7k/8/5K2/8/8/8/8/3Q4 w - - 0 1")
        
        # Enable draw prevention
        self.search.draw_prevention_enabled = True
        
        # Set up stalemate possibility
        stalemate_move = chess.Move.from_uci("d1d7")  # Queen to d7 is stalemate
        
        # Get available moves
        legal_moves = list(board.legal_moves)
        self.assertGreater(len(legal_moves), 1, "Test position should have multiple legal moves")
        
        # Find move
        move = self.search.iterative_deepening_search(board, board.turn)
        
        # Verify we got a valid move
        self.assertIsNotNone(move, "Search should return a valid move")
        if move:  # Add type guard for move
            # Verify we didn't choose stalemate
            self.assertNotEqual(move, stalemate_move)
            
            # Apply move and verify result
            board.push(move)
            self.assertFalse(board.is_stalemate())
            next_moves = list(board.legal_moves)
        self.assertGreater(len(next_moves), 0)
        
    def test_early_exit_on_clear_advantage(self):
        """Test early exit when a clear advantage is found."""
        # Position with a clear material advantage
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 1")
        
        # Set shorter time limit for test
        self.search.max_depth = 3  # Limit depth for quicker test
        self.time._test_allocated_time = 3.0  # Allow 3 seconds
        
        # Record timing
        start = time.time()
        move = self.search.iterative_deepening_search(board, board.turn)
        duration = time.time() - start
        
        # Verify quick exit and valid move
        self.assertLess(duration, self.time._test_allocated_time)
        self.assertNotEqual(move, chess.Move.null())
        
    def test_pv_line_stability(self):
        """Test early exit when PV line stabilizes."""
        # Starting position to ensure consistent search
        board = chess.Board()
        
        # Do a search that should stabilize quickly
        self.search.depth = 3  # Set fixed depth
        self.search.max_depth = 3
        self.search.nodes_searched = 0
        self.search.pv_move_stack = []
        
        # Do the search
        move = self.search.iterative_deepening_search(board, board.turn)
        self.assertNotEqual(move, chess.Move.null())
        
        # Verify PV line was tracked
        pv_stack = self.search.pv_move_stack
        self.assertGreaterEqual(len(pv_stack), 1)
        
        # Do another search at greater depth
        self.search.depth = 4
        self.search.max_depth = 4
        move = self.search.iterative_deepening_search(board, board.turn)
        self.assertNotEqual(move, chess.Move.null())
        
        # Verify PV line grows
        new_pv_stack = self.search.pv_move_stack
        self.assertGreater(len(new_pv_stack), len(pv_stack))
        
        # Check score stability
        if len(new_pv_stack) >= 2:
            scores = [entry['score'] for entry in new_pv_stack[-2:]]
            self.assertLess(abs(scores[1] - scores[0]), 0.5)
            
    def test_is_likely_draw(self):
        """Test draw detection in various positions."""
        # Test insufficient material
        board = chess.Board("8/8/8/3k4/8/3K4/8/8 w - - 0 1")
        self.assertTrue(self.rules.is_likely_draw(board))
        
        # Test blocked pawn structure
        board = chess.Board("8/pppppppp/PPPPPPPP/8/8/8/8/8 w - - 0 1")
        self.assertTrue(self.rules.is_likely_draw(board))
        
        # Test normal middlegame (should not be draw)
        board = chess.Board()
        self.assertFalse(self.rules.is_likely_draw(board))
        
    def test_draw_prevention_alternatives(self):
        """Test that draw prevention considers alternative moves."""
        # Position where one move leads to draw
        board = chess.Board("7k/8/7K/8/8/8/8/R7 w - - 0 1")
        
        # First get move without draw prevention
        self.search.draw_prevention_enabled = False
        move1: Optional[chess.Move] = None
        
        # Try a few moves to find one that leads to a draw
        for candidate in board.legal_moves:
            test_board = board.copy()
            test_board.push(candidate)
            if self.rules.is_likely_draw(test_board):
                move1 = candidate
                break
            
        self.assertIsNotNone(move1, "Should find a move leading to draw")
        
        # Now with draw prevention
        self.search.draw_prevention_enabled = True
        move2 = self.search.iterative_deepening_search(board, board.turn)
        
        # Moves should be different when draw prevention is enabled
        # Only compare if we found a drawing move
        if move1:
            self.assertNotEqual(move1, move2)

if __name__ == '__main__':
    unittest.main()
