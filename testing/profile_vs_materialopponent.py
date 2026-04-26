#!/usr/bin/env python3
"""
Deep Profiling Comparison: v7p3r Phase 1 vs MaterialOpponent

Goal: Find the 11K NPS gap (Phase 1: 24K vs MaterialOpponent: 35K)

This will instrument both engines and compare:
1. Evaluation function calls and timing
2. Quiescence search overhead  
3. Move generation and ordering
4. TT probe/store operations
5. Board make/unmake operations
6. Overall search tree size (nodes searched)
"""

import sys
import os
import time
import chess
from typing import Dict, List

# Add paths for both engines
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, r'e:\Programming Stuff\Chess Engines\Opponent Chess Engines\opponent-chess-engines\src\MaterialOpponent')

class InstrumentedPhase1:
    """Wrapper to instrument v7p3r Phase 1 with profiling"""
    
    def __init__(self):
        from v7p3r import V7P3REngine
        self.engine = V7P3REngine()
        
        # Profiling counters
        self.eval_calls = 0
        self.eval_time = 0.0
        self.quiescence_calls = 0
        self.quiescence_time = 0.0
        self.move_ordering_calls = 0
        self.move_ordering_time = 0.0
        self.tt_probe_calls = 0
        self.tt_probe_time = 0.0
        self.tt_store_calls = 0
        self.tt_store_time = 0.0
        self.board_push_calls = 0
        self.board_push_time = 0.0
        self.board_pop_calls = 0
        self.board_pop_time = 0.0
        
        # Wrap critical methods
        self._wrap_methods()
    
    def _wrap_methods(self):
        """Wrap engine methods with profiling instrumentation"""
        # Wrap evaluation
        original_eval = self.engine._evaluate_position
        def profiled_eval(board):
            self.eval_calls += 1
            start = time.perf_counter()
            result = original_eval(board)
            self.eval_time += time.perf_counter() - start
            return result
        self.engine._evaluate_position = profiled_eval
        
        # Wrap quiescence
        original_quiescence = self.engine._quiescence_search
        def profiled_quiescence(board, alpha, beta, depth=0):
            self.quiescence_calls += 1
            start = time.perf_counter()
            result = original_quiescence(board, alpha, beta, depth)
            self.quiescence_time += time.perf_counter() - start
            return result
        self.engine._quiescence_search = profiled_quiescence
        
        # Wrap move ordering
        original_order = self.engine._order_moves_advanced
        def profiled_order(*args, **kwargs):
            self.move_ordering_calls += 1
            start = time.perf_counter()
            result = original_order(*args, **kwargs)
            self.move_ordering_time += time.perf_counter() - start
            return result
        self.engine._order_moves_advanced = profiled_order
    
    def search(self, board, time_limit):
        """Run search with profiling"""
        result = self.engine.search(board, time_limit=time_limit)
        # Force update stats after search
        self.engine.search_stats['nodes_searched'] = self.engine.nodes_searched
        return result
    
    def get_stats(self) -> Dict:
        """Get profiling statistics"""
        return {
            'eval_calls': self.eval_calls,
            'eval_time': self.eval_time,
            'eval_avg_ms': (self.eval_time / self.eval_calls * 1000) if self.eval_calls > 0 else 0,
            'quiescence_calls': self.quiescence_calls,
            'quiescence_time': self.quiescence_time,
            'quiescence_avg_ms': (self.quiescence_time / self.quiescence_calls * 1000) if self.quiescence_calls > 0 else 0,
            'move_ordering_calls': self.move_ordering_calls,
            'move_ordering_time': self.move_ordering_time,
            'move_ordering_avg_ms': (self.move_ordering_time / self.move_ordering_calls * 1000) if self.move_ordering_calls > 0 else 0,
            'nodes_searched': self.engine.search_stats.get('nodes_searched', 0),
        }

class InstrumentedMaterialOpponent:
    """Wrapper to instrument MaterialOpponent with profiling"""
    
    def __init__(self):
        from material_opponent import MaterialOpponent
        self.engine = MaterialOpponent(max_depth=10)
        
        # Profiling counters
        self.eval_calls = 0
        self.eval_time = 0.0
        self.quiescence_calls = 0
        self.quiescence_time = 0.0
        self.move_ordering_calls = 0
        self.move_ordering_time = 0.0
        self.tt_probe_calls = 0
        self.tt_probe_time = 0.0
        self.tt_store_calls = 0
        self.tt_store_time = 0.0
        
        # Wrap critical methods
        self._wrap_methods()
    
    def _wrap_methods(self):
        """Wrap engine methods with profiling instrumentation"""
        # Wrap evaluation
        original_eval = self.engine._evaluate_material
        def profiled_eval(board):
            self.eval_calls += 1
            start = time.perf_counter()
            result = original_eval(board)
            self.eval_time += time.perf_counter() - start
            return result
        self.engine._evaluate_material = profiled_eval
        
        # Wrap quiescence
        original_quiescence = self.engine._quiescence_search
        def profiled_quiescence(board, alpha, beta, depth=0):
            self.quiescence_calls += 1
            start = time.perf_counter()
            result = original_quiescence(board, alpha, beta, depth)
            self.quiescence_time += time.perf_counter() - start
            return result
        self.engine._quiescence_search = profiled_quiescence
        
        # Wrap move ordering
        original_order = self.engine._order_moves
        def profiled_order(board, moves, ply, tt_move=None):
            self.move_ordering_calls += 1
            start = time.perf_counter()
            result = original_order(board, moves, ply, tt_move)
            self.move_ordering_time += time.perf_counter() - start
            return result
        self.engine._order_moves = profiled_order
        
        # Wrap TT probe
        original_probe = self.engine._probe_tt
        def profiled_probe(zobrist_key, depth, alpha, beta):
            self.tt_probe_calls += 1
            start = time.perf_counter()
            result = original_probe(zobrist_key, depth, alpha, beta)
            self.tt_probe_time += time.perf_counter() - start
            return result
        self.engine._probe_tt = profiled_probe
        
        # Wrap TT store
        original_store = self.engine._store_tt_entry
        def profiled_store(zobrist_key, depth, value, node_type, best_move):
            self.tt_store_calls += 1
            start = time.perf_counter()
            original_store(zobrist_key, depth, value, node_type, best_move)
            self.tt_store_time += time.perf_counter() - start
        self.engine._store_tt_entry = profiled_store
    
    def search(self, board, time_limit):
        """Run search with profiling"""
        self.engine.board = board.copy()
        self.engine.time_limit = time_limit  # Set fixed time limit
        self.engine.max_depth = 10  # Allow deep search
        return self.engine.get_best_move(time_left=0)  # Pass 0 to disable time management
    
    def get_stats(self) -> Dict:
        """Get profiling statistics"""
        return {
            'eval_calls': self.eval_calls,
            'eval_time': self.eval_time,
            'eval_avg_ms': (self.eval_time / self.eval_calls * 1000) if self.eval_calls > 0 else 0,
            'quiescence_calls': self.quiescence_calls,
            'quiescence_time': self.quiescence_time,
            'quiescence_avg_ms': (self.quiescence_time / self.quiescence_calls * 1000) if self.quiescence_calls > 0 else 0,
            'move_ordering_calls': self.move_ordering_calls,
            'move_ordering_time': self.move_ordering_time,
            'move_ordering_avg_ms': (self.move_ordering_time / self.move_ordering_calls * 1000) if self.move_ordering_calls > 0 else 0,
            'tt_probe_calls': self.tt_probe_calls,
            'tt_probe_time': self.tt_probe_time,
            'tt_probe_avg_ms': (self.tt_probe_time / self.tt_probe_calls * 1000) if self.tt_probe_calls > 0 else 0,
            'tt_store_calls': self.tt_store_calls,
            'tt_store_time': self.tt_store_time,
            'tt_store_avg_ms': (self.tt_store_time / self.tt_store_calls * 1000) if self.tt_store_calls > 0 else 0,
            'nodes_searched': self.engine.nodes_searched,
        }

def main():
    print("""
================================================================================
DEEP PROFILING: v7p3r Phase 1 vs MaterialOpponent
================================================================================
Goal: Find the 11K NPS performance gap (Phase 1: 24K vs MaterialOpponent: 35K)

Profiling both engines on the same position to identify bottlenecks:
- Evaluation calls and timing
- Quiescence search overhead
- Move ordering complexity
- TT operations
- Node count comparison

Test Position: Standard middlegame tactical position
Time Limit: 5 seconds each
================================================================================
""")
    
    # Test position - complex middlegame
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
    board = chess.Board(fen)
    
    print(f"Position: {fen}")
    print(f"Legal moves: {board.legal_moves.count()}\n")
    
    # Test Phase 1
    print("=" * 80)
    print("TESTING v7p3r Phase 1...")
    print("=" * 80)
    
    phase1 = InstrumentedPhase1()
    start = time.time()
    move_phase1 = phase1.search(board, time_limit=5.0)
    time_phase1 = time.time() - start
    stats_phase1 = phase1.get_stats()
    
    print(f"\nPhase 1 completed in {time_phase1:.2f}s")
    print(f"Best move: {move_phase1}")
    print(f"Nodes: {stats_phase1['nodes_searched']:,}")
    print(f"NPS: {stats_phase1['nodes_searched']/time_phase1:,.0f}")
    
    # Test MaterialOpponent
    print("\n" + "=" * 80)
    print("TESTING MaterialOpponent...")
    print("=" * 80)
    
    material = InstrumentedMaterialOpponent()
    start = time.time()
    move_material = material.search(board, time_limit=5.0)
    time_material = time.time() - start
    stats_material = material.get_stats()
    
    print(f"\nMaterialOpponent completed in {time_material:.2f}s")
    print(f"Best move: {move_material}")
    print(f"Nodes: {stats_material['nodes_searched']:,}")
    print(f"NPS: {stats_material['nodes_searched']/time_material:,.0f}")
    
    # Comparative Analysis
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    print(f"\n{'Component':<30} {'Phase 1':<20} {'MaterialOpp':<20} {'Difference':<20}")
    print("─" * 90)
    
    # NPS comparison
    nps_phase1 = stats_phase1['nodes_searched'] / time_phase1 if time_phase1 > 0 and stats_phase1['nodes_searched'] > 0 else 0
    nps_material = stats_material['nodes_searched'] / time_material if time_material > 0 and stats_material['nodes_searched'] > 0 else 0
    nps_diff = nps_material - nps_phase1
    nps_pct = (nps_diff / nps_phase1) * 100 if nps_phase1 > 0 else 0
    print(f"{'NPS':<30} {nps_phase1:>19,.0f} {nps_material:>19,.0f} {nps_diff:>+18,.0f} ({nps_pct:+.1f}%)")
    
    # Nodes searched
    nodes_diff = stats_material['nodes_searched'] - stats_phase1['nodes_searched']
    nodes_pct = (nodes_diff / stats_phase1['nodes_searched']) * 100 if stats_phase1['nodes_searched'] > 0 else 0
    print(f"{'Nodes Searched':<30} {stats_phase1['nodes_searched']:>19,} {stats_material['nodes_searched']:>19,} {nodes_diff:>+18,} ({nodes_pct:+.1f}%)")
    
    # Evaluation calls
    eval_diff = stats_material['eval_calls'] - stats_phase1['eval_calls']
    eval_pct = (eval_diff / stats_phase1['eval_calls']) * 100 if stats_phase1['eval_calls'] > 0 else 0
    print(f"{'Evaluation Calls':<30} {stats_phase1['eval_calls']:>19,} {stats_material['eval_calls']:>19,} {eval_diff:>+18,} ({eval_pct:+.1f}%)")
    
    # Evaluation time
    print(f"{'Evaluation Total Time':<30} {stats_phase1['eval_time']:>18.3f}s {stats_material['eval_time']:>18.3f}s {stats_material['eval_time']-stats_phase1['eval_time']:>+17.3f}s")
    print(f"{'Evaluation Avg Time':<30} {stats_phase1['eval_avg_ms']:>17.4f}ms {stats_material['eval_avg_ms']:>17.4f}ms {stats_material['eval_avg_ms']-stats_phase1['eval_avg_ms']:>+16.4f}ms")
    
    # Quiescence calls
    qsearch_diff = stats_material['quiescence_calls'] - stats_phase1['quiescence_calls']
    qsearch_pct = (qsearch_diff / stats_phase1['quiescence_calls']) * 100 if stats_phase1['quiescence_calls'] > 0 else 0
    print(f"{'Quiescence Calls':<30} {stats_phase1['quiescence_calls']:>19,} {stats_material['quiescence_calls']:>19,} {qsearch_diff:>+18,} ({qsearch_pct:+.1f}%)")
    
    # Quiescence time
    print(f"{'Quiescence Total Time':<30} {stats_phase1['quiescence_time']:>18.3f}s {stats_material['quiescence_time']:>18.3f}s {stats_material['quiescence_time']-stats_phase1['quiescence_time']:>+17.3f}s")
    print(f"{'Quiescence Avg Time':<30} {stats_phase1['quiescence_avg_ms']:>17.4f}ms {stats_material['quiescence_avg_ms']:>17.4f}ms {stats_material['quiescence_avg_ms']-stats_phase1['quiescence_avg_ms']:>+16.4f}ms")
    
    # Move ordering
    ordering_diff = stats_material['move_ordering_calls'] - stats_phase1['move_ordering_calls']
    ordering_pct = (ordering_diff / stats_phase1['move_ordering_calls']) * 100 if stats_phase1['move_ordering_calls'] > 0 else 0
    print(f"{'Move Ordering Calls':<30} {stats_phase1['move_ordering_calls']:>19,} {stats_material['move_ordering_calls']:>19,} {ordering_diff:>+18,} ({ordering_pct:+.1f}%)")
    
    # Move ordering time
    print(f"{'Move Ordering Total Time':<30} {stats_phase1['move_ordering_time']:>18.3f}s {stats_material['move_ordering_time']:>18.3f}s {stats_material['move_ordering_time']-stats_phase1['move_ordering_time']:>+17.3f}s")
    print(f"{'Move Ordering Avg Time':<30} {stats_phase1['move_ordering_avg_ms']:>17.4f}ms {stats_material['move_ordering_avg_ms']:>17.4f}ms {stats_material['move_ordering_avg_ms']-stats_phase1['move_ordering_avg_ms']:>+16.4f}ms")
    
    # TT operations (MaterialOpponent only - v7p3r uses chess library's TT)
    print(f"\n{'TT Probe Calls (MaterialOpp)':<30} {'':<20} {stats_material['tt_probe_calls']:>19,}")
    print(f"{'TT Probe Avg Time':<30} {'':<20} {stats_material['tt_probe_avg_ms']:>17.4f}ms")
    print(f"{'TT Store Calls (MaterialOpp)':<30} {'':<20} {stats_material['tt_store_calls']:>19,}")
    print(f"{'TT Store Avg Time':<30} {'':<20} {stats_material['tt_store_avg_ms']:>17.4f}ms")
    
    # Performance percentage breakdown
    print("\n" + "=" * 80)
    print("TIME DISTRIBUTION")
    print("=" * 80)
    
    print(f"\n{'Component':<30} {'Phase 1 %':<20} {'MaterialOpp %':<20}")
    print("─" * 70)
    
    phase1_eval_pct = (stats_phase1['eval_time'] / time_phase1) * 100
    material_eval_pct = (stats_material['eval_time'] / time_material) * 100
    print(f"{'Evaluation':<30} {phase1_eval_pct:>18.1f}% {material_eval_pct:>18.1f}%")
    
    phase1_quiescence_pct = (stats_phase1['quiescence_time'] / time_phase1) * 100
    material_quiescence_pct = (stats_material['quiescence_time'] / time_material) * 100
    print(f"{'Quiescence':<30} {phase1_quiescence_pct:>18.1f}% {material_quiescence_pct:>18.1f}%")
    
    phase1_ordering_pct = (stats_phase1['move_ordering_time'] / time_phase1) * 100
    material_ordering_pct = (stats_material['move_ordering_time'] / time_material) * 100
    print(f"{'Move Ordering':<30} {phase1_ordering_pct:>18.1f}% {material_ordering_pct:>18.1f}%")
    
    material_tt_pct = ((stats_material['tt_probe_time'] + stats_material['tt_store_time']) / time_material) * 100
    print(f"{'TT Operations':<30} {'N/A':<20} {material_tt_pct:>18.1f}%")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS & BOTTLENECKS")
    print("=" * 80)
    
    findings = []
    
    if phase1_eval_pct > material_eval_pct + 5:
        findings.append(f"🔴 Evaluation is {phase1_eval_pct - material_eval_pct:.1f}% slower in Phase 1")
    
    if phase1_quiescence_pct > material_quiescence_pct + 5:
        findings.append(f"🔴 Quiescence is {phase1_quiescence_pct - material_quiescence_pct:.1f}% slower in Phase 1")
    
    if phase1_ordering_pct > material_ordering_pct + 5:
        findings.append(f"🔴 Move ordering is {phase1_ordering_pct - material_ordering_pct:.1f}% slower in Phase 1")
    
    if nps_material > nps_phase1 * 1.2:
        findings.append(f"🔴 Overall NPS gap: {nps_pct:.1f}% slower in Phase 1")
    
    if findings:
        for finding in findings:
            print(f"\n{finding}")
    else:
        print("\n✓ Performance is comparable - no major bottlenecks identified")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
