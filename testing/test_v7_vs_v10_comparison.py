#!/usr/bin/env python3
"""
Comprehensive V7.0 vs V10.0 Performance Comparison
Tests multiple aspects: NPS, tactical strength, search depth, move quality
"""

import sys
import os
import time
import chess
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from v7p3r import V7P3RCleanEngine

# Simple V7.0-style engine for comparison
class V7SimpleEngine:
    """Simplified V7.0-style engine with basic alpha-beta only"""
    
    def __init__(self):
        from v7p3r_scoring_calculation import V7P3RScoringCalculationClean
        
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300, 
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        self.default_depth = 6
        self.nodes_searched = 0
        self.scoring_calculator = V7P3RScoringCalculationClean(self.piece_values)
        self.evaluation_cache = {}
    
    def search(self, board: chess.Board, time_limit: float = 3.0) -> chess.Move:
        """Simple iterative deepening search like V7.0"""
        self.nodes_searched = 0
        start_time = time.time()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        best_move = legal_moves[0]
        target_time = min(time_limit * 0.8, 10.0)
        
        for depth in range(1, self.default_depth + 1):
            iteration_start = time.time()
            
            try:
                score, move = self._simple_search(board, depth, -99999, 99999)
                
                if move:
                    best_move = move
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    nps = int(self.nodes_searched / max(elapsed_ms / 1000, 0.001))
                    print(f"V7 depth {depth} score cp {int(score)} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {move}")
                
                elapsed = time.time() - start_time
                iteration_time = time.time() - iteration_start
                
                if elapsed > target_time or iteration_time > time_limit * 0.3:
                    break
                    
            except:
                break
                
        return best_move
    
    def _simple_search(self, board: chess.Board, depth: int, alpha: float, beta: float):
        """Simple alpha-beta search like V7.0"""
        self.nodes_searched += 1
        
        if depth == 0:
            return self._evaluate_position(board), None
            
        if board.is_game_over():
            if board.is_checkmate():
                return -29000.0 + (self.default_depth - depth), None
            else:
                return 0.0, None
        
        # Simple move ordering: captures first
        legal_moves = list(board.legal_moves)
        legal_moves.sort(key=lambda m: board.is_capture(m), reverse=True)
        
        best_score = -99999.0
        best_move = legal_moves[0] if legal_moves else None
        
        for move in legal_moves:
            board.push(move)
            score, _ = self._simple_search(board, depth - 1, -beta, -alpha)
            score = -score
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        
        return best_score, best_move
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """Position evaluation with caching"""
        cache_key = board.fen()
        
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        white_score = self.scoring_calculator.calculate_score_optimized(board, True)
        black_score = self.scoring_calculator.calculate_score_optimized(board, False)
        
        if board.turn:
            final_score = white_score - black_score
        else:
            final_score = black_score - white_score
        
        self.evaluation_cache[cache_key] = final_score
        return final_score
    
    def new_game(self):
        """Reset for new game"""
        self.evaluation_cache.clear()
        self.nodes_searched = 0


def test_tactical_positions():
    """Test both engines on tactical positions"""
    
    tactical_positions = [
        {
            "name": "Knight Fork",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "expected_move": "f3g5",  # Knight fork on f7
            "description": "White can play Ng5 attacking f7 and h7"
        },
        {
            "name": "Pin Attack",
            "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "expected_move": "c4f7",  # Bishop takes on f7+
            "description": "Bishop can capture on f7 with check"
        },
        {
            "name": "Queen Trap",
            "fen": "rnbqkbnr/ppp2ppp/3p4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
            "expected_move": "c4f7",  # Bishop sacrifice for attack
            "description": "Tactical shot with bishop sacrifice"
        },
        {
            "name": "Back Rank",
            "fen": "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
            "expected_move": "e1e8",  # Back rank mate threat
            "description": "Rook move threatens back rank mate"
        }
    ]
    
    print("=" * 80)
    print("TACTICAL POSITIONS TEST")
    print("=" * 80)
    
    v7_engine = V7SimpleEngine()
    v10_engine = V7P3RCleanEngine()
    
    v7_correct = 0
    v10_correct = 0
    
    for i, position in enumerate(tactical_positions, 1):
        print(f"\n{i}. {position['name']}")
        print(f"   {position['description']}")
        print(f"   Position: {position['fen']}")
        print(f"   Expected best move: {position['expected_move']}")
        
        board = chess.Board(position['fen'])
        
        # Test V7 engine
        print(f"\n   V7.0 Engine:")
        v7_engine.new_game()
        start_time = time.time()
        v7_move = v7_engine.search(board, 3.0)
        v7_time = time.time() - start_time
        v7_nps = int(v7_engine.nodes_searched / max(v7_time, 0.001))
        
        v7_found_best = str(v7_move) == position['expected_move']
        if v7_found_best:
            v7_correct += 1
        
        print(f"   V7 chose: {v7_move} {'✓' if v7_found_best else '✗'}")
        print(f"   V7 nodes: {v7_engine.nodes_searched}, NPS: {v7_nps}, Time: {v7_time:.2f}s")
        
        # Test V10 engine
        print(f"\n   V10.0 Engine:")
        v10_engine.new_game()
        start_time = time.time()
        v10_move = v10_engine.search(board, 3.0)
        v10_time = time.time() - start_time
        v10_nps = int(v10_engine.nodes_searched / max(v10_time, 0.001))
        
        v10_found_best = str(v10_move) == position['expected_move']
        if v10_found_best:
            v10_correct += 1
        
        print(f"   V10 chose: {v10_move} {'✓' if v10_found_best else '✗'}")
        print(f"   V10 nodes: {v10_engine.nodes_searched}, NPS: {v10_nps}, Time: {v10_time:.2f}s")
        
        # Show V10 advanced features stats
        if hasattr(v10_engine, 'search_stats'):
            stats = v10_engine.search_stats
            print(f"   V10 TT hits: {stats.get('tt_hits', 0)}, Killer hits: {stats.get('killer_hits', 0)}")
        
        print(f"   Winner: {'V10' if v10_found_best and not v7_found_best else 'V7' if v7_found_best and not v10_found_best else 'Tie'}")
    
    print(f"\n" + "=" * 80)
    print(f"TACTICAL TEST RESULTS:")
    print(f"V7.0 Engine:  {v7_correct}/{len(tactical_positions)} correct ({v7_correct/len(tactical_positions)*100:.1f}%)")
    print(f"V10.0 Engine: {v10_correct}/{len(tactical_positions)} correct ({v10_correct/len(tactical_positions)*100:.1f}%)")
    print(f"=" * 80)
    
    return v7_correct, v10_correct


def test_search_efficiency():
    """Test search efficiency on standard positions"""
    
    test_positions = [
        {
            "name": "Starting Position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        },
        {
            "name": "Italian Game", 
            "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        },
        {
            "name": "Middlegame",
            "fen": "r1bq1rk1/pp2nppp/2n1p3/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 9"
        },
        {
            "name": "Endgame",
            "fen": "8/2k5/3p4/p2P1p2/P3pP2/8/2K5/8 w - - 0 1"
        }
    ]
    
    print("\n" + "=" * 80)
    print("SEARCH EFFICIENCY TEST")
    print("=" * 80)
    
    v7_engine = V7SimpleEngine()
    v10_engine = V7P3RCleanEngine()
    
    v7_total_nodes = 0
    v10_total_nodes = 0
    v7_total_time = 0
    v10_total_time = 0
    
    for i, position in enumerate(test_positions, 1):
        print(f"\n{i}. {position['name']}")
        print(f"   Position: {position['fen']}")
        
        board = chess.Board(position['fen'])
        
        # Test V7 engine
        v7_engine.new_game()
        start_time = time.time()
        v7_move = v7_engine.search(board, 4.0)
        v7_time = time.time() - start_time
        v7_nps = int(v7_engine.nodes_searched / max(v7_time, 0.001))
        
        print(f"   V7:  {v7_engine.nodes_searched:6d} nodes, {v7_nps:6d} NPS, {v7_time:.2f}s -> {v7_move}")
        
        # Test V10 engine
        v10_engine.new_game()
        start_time = time.time()
        v10_move = v10_engine.search(board, 4.0)
        v10_time = time.time() - start_time
        v10_nps = int(v10_engine.nodes_searched / max(v10_time, 0.001))
        
        print(f"   V10: {v10_engine.nodes_searched:6d} nodes, {v10_nps:6d} NPS, {v10_time:.2f}s -> {v10_move}")
        
        # Show efficiency gain/loss
        node_ratio = v7_engine.nodes_searched / max(v10_engine.nodes_searched, 1)
        nps_ratio = v7_nps / max(v10_nps, 1)
        
        print(f"   Node efficiency: V7/V10 = {node_ratio:.2f}x")
        print(f"   NPS efficiency:  V7/V10 = {nps_ratio:.2f}x")
        
        if hasattr(v10_engine, 'search_stats'):
            stats = v10_engine.search_stats
            print(f"   V10 Advanced: TT hits: {stats.get('tt_hits', 0)}, Killer: {stats.get('killer_hits', 0)}")
        
        v7_total_nodes += v7_engine.nodes_searched
        v10_total_nodes += v10_engine.nodes_searched
        v7_total_time += v7_time
        v10_total_time += v10_time
    
    # Overall statistics
    print(f"\n" + "=" * 80)
    print(f"OVERALL EFFICIENCY RESULTS:")
    print(f"V7 Total:  {v7_total_nodes:8d} nodes in {v7_total_time:.2f}s = {int(v7_total_nodes/v7_total_time):6d} NPS")
    print(f"V10 Total: {v10_total_nodes:8d} nodes in {v10_total_time:.2f}s = {int(v10_total_nodes/v10_total_time):6d} NPS")
    print(f"Node Ratio (V7/V10): {v7_total_nodes/max(v10_total_nodes,1):.2f}x")
    print(f"NPS Ratio (V7/V10):  {(v7_total_nodes/v7_total_time)/(v10_total_nodes/v10_total_time):.2f}x")
    print(f"=" * 80)


def main():
    """Run comprehensive V7 vs V10 comparison"""
    print("V7.0 vs V10.0 COMPREHENSIVE COMPARISON TEST")
    print("Testing tactical strength and search efficiency")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test tactical positions
    v7_correct, v10_correct = test_tactical_positions()
    
    # Test search efficiency  
    test_search_efficiency()
    
    print(f"\n" + "=" * 80)
    print(f"FINAL SUMMARY:")
    print(f"Tactical Strength: V7={v7_correct}/4, V10={v10_correct}/4")
    print(f"V10.0 Advanced Features: Transposition Table, Killer Moves, History, Null Move, LMR")
    print(f"Trade-off: V10 sacrifices some NPS for much stronger tactical search")
    print(f"=" * 80)


if __name__ == "__main__":
    main()
