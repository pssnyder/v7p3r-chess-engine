#!/usr/bin/env python3
"""
V9.4-Beta Engine Integration
Full engine implementation using v9.4-beta scoring for game testing vs v7.0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
import random
from typing import Optional
from v7p3r_scoring_calculation_v94_beta import V7P3RScoringCalculationV94Beta

class V7P3REngineV94Beta:
    """V9.4-beta chess engine for head-to-head testing vs v7.0"""
    
    def __init__(self):
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        self.scorer = V7P3RScoringCalculationV94Beta(self.piece_values)
        self.search_depth = 4
        self.nodes_searched = 0
        
    def search(self, board: chess.Board, time_limit: float = 2.0) -> Optional[chess.Move]:
        """
        Search for the best move in the position
        """
        self.nodes_searched = 0
        start_time = time.time()
        
        best_move = None
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        # Quick random move if very short time
        if time_limit < 0.1:
            return random.choice(legal_moves)
        
        # Iterative deepening search
        for depth in range(1, min(self.search_depth + 1, 6)):
            if time.time() - start_time > time_limit * 0.8:  # Use 80% of time
                break
                
            current_best = self._search_depth(board, depth, float('-inf'), float('inf'), board.turn)
            
            if current_best[1] is not None:  # Valid move found
                best_move = current_best[1]
                best_score = current_best[0]
                
                # Output search info
                elapsed = time.time() - start_time
                nps = int(self.nodes_searched / max(elapsed, 0.001))
                score_cp = int(best_score * 100)  # Convert to centipawns
                
                print(f"info depth {depth} score cp {score_cp} nodes {self.nodes_searched} "
                      f"time {int(elapsed * 1000)} nps {nps} pv {best_move}")
        
        return best_move
    
    def _search_depth(self, board: chess.Board, depth: int, alpha: float, beta: float, 
                     maximizing_player: bool, root_call: bool = True):
        """
        Minimax search with alpha-beta pruning
        """
        self.nodes_searched += 1
        
        if depth == 0 or board.is_game_over():
            score = self._evaluate_position(board)
            return (score, None)
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return (self._evaluate_position(board), None)
        
        best_move = None
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval_score, _ = self._search_depth(board, depth - 1, alpha, beta, False, False)
                board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
                    
            return (max_eval, best_move)
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score, _ = self._search_depth(board, depth - 1, alpha, beta, True, False)
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
                    
            return (min_eval, best_move)
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate the current position using v9.4-beta scoring
        """
        if board.is_checkmate():
            return -10000.0 if board.turn == chess.WHITE else 10000.0
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        
        white_score = self.scorer.calculate_score_optimized(board, chess.WHITE)
        black_score = self.scorer.calculate_score_optimized(board, chess.BLACK)
        
        return white_score - black_score

def run_game_vs_engine(v94_engine, opponent_engine, time_per_move=1.0):
    """
    Run a single game between v9.4-beta and another engine
    """
    board = chess.Board()
    move_count = 0
    
    print(f"Starting game: V9.4-beta vs Opponent")
    print(f"Time per move: {time_per_move} seconds")
    print("=" * 50)
    
    while not board.is_game_over() and move_count < 100:  # Max 100 moves
        move_count += 1
        
        if board.turn == chess.WHITE:
            # V9.4-beta plays White
            print(f"Move {move_count}: V9.4-beta (White) thinking...")
            move = v94_engine.search(board, time_per_move)
            player = "V9.4-beta"
        else:
            # Opponent plays Black  
            print(f"Move {move_count}: Opponent (Black) thinking...")
            move = opponent_engine.search(board, time_per_move)
            player = "Opponent"
        
        if move is None:
            print(f"No legal move found for {player}")
            break
            
        board.push(move)
        print(f"{player} plays: {move}")
        
        # Show position evaluation from v9.4 perspective
        evaluation = v94_engine._evaluate_position(board)
        print(f"V9.4-beta evaluation: {evaluation:.2f}")
        print()
    
    # Game result
    result = board.result()
    print("=" * 50)
    print(f"Game finished: {result}")
    
    if result == "1-0":
        return "v94_win"
    elif result == "0-1":
        return "opponent_win"
    else:
        return "draw"

def simple_opponent_engine():
    """
    Create a simple opponent engine for testing (can be replaced with v7.0)
    """
    class SimpleEngine:
        def search(self, board, time_limit=1.0):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
            # Just return a random legal move for now
            return random.choice(legal_moves)
    
    return SimpleEngine()

def run_match_series(num_games=5):
    """
    Run a series of games for testing
    """
    print("V7P3R v9.4-Beta Game Testing")
    print("=" * 60)
    
    v94_engine = V7P3REngineV94Beta()
    opponent = simple_opponent_engine()
    
    results = []
    v94_wins = 0
    opponent_wins = 0
    draws = 0
    
    for game_num in range(1, num_games + 1):
        print(f"\nGAME {game_num} of {num_games}")
        print("-" * 40)
        
        result = run_game_vs_engine(v94_engine, opponent, time_per_move=0.5)
        results.append(result)
        
        if result == "v94_win":
            v94_wins += 1
        elif result == "opponent_win":
            opponent_wins += 1
        else:
            draws += 1
        
        print(f"Game {game_num} result: {result}")
        print(f"Current score - V9.4: {v94_wins}, Opponent: {opponent_wins}, Draws: {draws}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("MATCH SERIES RESULTS")
    print("=" * 60)
    print(f"Games played: {num_games}")
    print(f"V9.4-beta wins: {v94_wins}")
    print(f"Opponent wins: {opponent_wins}")
    print(f"Draws: {draws}")
    
    win_rate = v94_wins / num_games
    print(f"V9.4-beta win rate: {win_rate:.1%}")
    
    if win_rate >= 0.6:
        print("\n🎉 V9.4-beta shows strong performance!")
        print("✓ Ready for final v9.4 release")
    elif win_rate >= 0.4:
        print("\n⚡ V9.4-beta shows decent performance")
        print("→ Consider minor adjustments")
    else:
        print("\n❌ V9.4-beta needs improvement")
        print("→ Review evaluation or search")
    
    return results

def main():
    """
    Main testing function
    """
    print("V9.4-Beta Engine Integration Test")
    print("=" * 50)
    
    # Quick engine validation
    print("Testing engine initialization...")
    engine = V7P3REngineV94Beta()
    
    board = chess.Board()
    move = engine.search(board, 1.0)
    
    if move:
        print(f"✓ Engine working - suggests: {move}")
    else:
        print("❌ Engine failed to find move")
        return False
    
    # Run match series
    print("\nRunning match series...")
    results = run_match_series(3)  # Start with 3 games for quick testing
    
    print("\n🎯 Next Steps:")
    print("1. Replace simple opponent with actual v7.0 engine")
    print("2. Run longer match series (20+ games)")
    print("3. Analyze specific game positions")
    print("4. Finalize v9.4 for v10.0 release")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
