"""
Component-level profiling for V7P3R v19.2

This instruments the actual search to measure time spent in each component:
- Move generation
- Move ordering  
- Transposition table operations
- Evaluation
- Quiescence search
- Board make/unmake moves

Goal: Find the REAL bottleneck causing ~9k NPS instead of 50k-100k+ NPS
"""

import sys
import os
import time
import chess
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

class ProfiledEngine(V7P3REngine):
    """Instrumented engine that tracks time per component"""
    
    def __init__(self):
        super().__init__()
        self.profile_times = defaultdict(float)
        self.profile_counts = defaultdict(int)
    
    def _recursive_search(self, board, search_depth, alpha, beta, time_limit):
        """Override to add timing instrumentation"""
        self.nodes_searched += 1
        
        # Time check
        if hasattr(self, 'search_start_time') and self.nodes_searched % 1000 == 0:
            elapsed = time.time() - self.search_start_time
            if elapsed > time_limit:
                return self._evaluate_position(board), None
        
        # TT Probe - TIMED
        tt_start = time.time()
        tt_hit, tt_score, tt_move = self._probe_transposition_table(board, search_depth, int(alpha), int(beta))
        self.profile_times['tt_probe'] += time.time() - tt_start
        self.profile_counts['tt_probe'] += 1
        
        if tt_hit:
            return float(tt_score), tt_move
        
        # Terminal conditions
        if search_depth == 0:
            # Quiescence - TIMED
            q_start = time.time()
            score = self._quiescence_search(board, alpha, beta, 4)
            self.profile_times['quiescence'] += time.time() - q_start
            self.profile_counts['quiescence'] += 1
            return score, None
        
        # Move generation - TIMED
        movegen_start = time.time()
        legal_moves = list(board.legal_moves)
        self.profile_times['move_generation'] += time.time() - movegen_start
        self.profile_counts['move_generation'] += 1
        
        if not legal_moves:
            if board.is_check():
                score = -29000.0 + (self.default_depth - search_depth)
            else:
                score = 0.0
            return score, None
        
        # Null move pruning
        if (search_depth >= 3 and not board.is_check() and 
            self._has_non_pawn_pieces(board) and beta - alpha > 1):
            
            board.turn = not board.turn
            null_score, _ = self._recursive_search(board, search_depth - 2, -beta, -beta + 1, time_limit)
            null_score = -null_score
            board.turn = not board.turn
            
            if null_score >= beta:
                return null_score, None
        
        # Move ordering - TIMED
        order_start = time.time()
        ordered_moves = self._order_moves_advanced(board, legal_moves, search_depth, tt_move)
        self.profile_times['move_ordering'] += time.time() - order_start
        self.profile_counts['move_ordering'] += 1
        
        # Main search loop
        best_score = -99999.0
        best_move = None
        original_alpha = alpha
        moves_searched = 0
        
        for move in ordered_moves:
            # Make move - TIMED
            make_start = time.time()
            board.push(move)
            self.profile_times['board_make'] += time.time() - make_start
            self.profile_counts['board_make'] += 1
            
            # LMR
            reduction = self._calculate_lmr_reduction(move, moves_searched, search_depth, board)
            
            if reduction > 0:
                score, _ = self._recursive_search(board, search_depth - 1 - reduction, -beta, -alpha, time_limit)
                score = -score
                if score > alpha:
                    score, _ = self._recursive_search(board, search_depth - 1, -beta, -alpha, time_limit)
                    score = -score
            else:
                score, _ = self._recursive_search(board, search_depth - 1, -beta, -alpha, time_limit)
                score = -score
            
            # Unmake move - TIMED
            unmake_start = time.time()
            board.pop()
            self.profile_times['board_unmake'] += time.time() - unmake_start
            self.profile_counts['board_unmake'] += 1
            
            moves_searched += 1
            
            if best_move is None or score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            if alpha >= beta:
                if not board.is_capture(move):
                    self.killer_moves.store_killer(move, search_depth)
                    self.history_heuristic.update_history(move, search_depth)
                    self.search_stats['killer_hits'] += 1
                break
        
        # TT store - TIMED
        tt_store_start = time.time()
        self._store_transposition_table(board, search_depth, int(best_score), best_move, int(original_alpha), int(beta))
        self.profile_times['tt_store'] += time.time() - tt_store_start
        self.profile_counts['tt_store'] += 1
        
        return best_score, best_move
    
    def _evaluate_position(self, board):
        """Override to time evaluation calls"""
        eval_start = time.time()
        result = super()._evaluate_position(board)
        self.profile_times['evaluation'] += time.time() - eval_start
        self.profile_counts['evaluation'] += 1
        return result
    
    def print_profile_report(self, total_time):
        """Print detailed profiling report"""
        print("\n" + "="*80)
        print("COMPONENT-LEVEL PROFILING REPORT")
        print("="*80)
        
        # Calculate percentages
        components = [
            ('move_generation', 'Move Generation'),
            ('move_ordering', 'Move Ordering'),
            ('tt_probe', 'TT Probe'),
            ('tt_store', 'TT Store'),
            ('evaluation', 'Position Evaluation'),
            ('quiescence', 'Quiescence Search'),
            ('board_make', 'Board Make Move'),
            ('board_unmake', 'Board Unmake Move'),
        ]
        
        total_profiled = sum(self.profile_times.values())
        
        print(f"\nTotal search time: {total_time:.3f}s")
        print(f"Total profiled time: {total_profiled:.3f}s ({100*total_profiled/total_time:.1f}%)")
        print(f"Unprofiled overhead: {total_time - total_profiled:.3f}s ({100*(total_time-total_profiled)/total_time:.1f}%)\n")
        
        print(f"{'Component':<25} {'Time (s)':<12} {'% Total':<10} {'Calls':<12} {'Avg (ms)':<10}")
        print("-" * 80)
        
        for key, name in components:
            if key in self.profile_times:
                time_spent = self.profile_times[key]
                pct = 100 * time_spent / total_time
                calls = self.profile_counts[key]
                avg_ms = 1000 * time_spent / calls if calls > 0 else 0
                print(f"{name:<25} {time_spent:>10.3f}s  {pct:>7.1f}%  {calls:>10,}  {avg_ms:>8.4f}ms")
        
        print("-" * 80)
        print(f"{'TOTAL PROFILED':<25} {total_profiled:>10.3f}s  {100*total_profiled/total_time:>7.1f}%")
        print("="*80)
        
        # Identify bottlenecks
        print("\n🔥 BOTTLENECK ANALYSIS:")
        print("-" * 80)
        
        sorted_components = sorted([(v, k) for k, v in self.profile_times.items()], reverse=True)
        
        for i, (time_spent, key) in enumerate(sorted_components[:5], 1):
            name = dict(components).get(key, key)
            pct = 100 * time_spent / total_time
            print(f"{i}. {name}: {time_spent:.3f}s ({pct:.1f}% of total)")
        
        print("="*80)


def main():
    print("="*80)
    print("V7P3R v19.2 COMPONENT-LEVEL PROFILING")
    print("="*80)
    
    engine = ProfiledEngine()
    board = chess.Board()
    
    # Complex middlegame position
    board.set_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
    
    print(f"\nTest Position: {board.fen()}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    print("\nSearching for 5 seconds with component-level instrumentation...\n")
    
    # Search
    start = time.time()
    result = engine.search(board, time_limit=5.0)
    elapsed = time.time() - start
    
    # Results
    print(f"\n{'='*80}")
    print(f"SEARCH RESULTS:")
    print(f"  Best move: {result}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Nodes: {engine.nodes_searched:,}")
    print(f"  NPS: {engine.nodes_searched/elapsed:,.0f}")
    
    # Print detailed profile
    engine.print_profile_report(elapsed)


if __name__ == '__main__':
    main()
