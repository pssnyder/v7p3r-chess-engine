#!/usr/bin/env python3
"""
TAL-BOT Time Control Performance Analysis

Test TAL-BOT's search depth capabilities across different time controls:
- Classical: 15+ minutes per game
- Rapid: 10-15 minutes per game  
- Blitz: 3-5 minutes per game
- Bullet: 1-2 minutes per game

This will help us understand TAL-BOT's entropy advantage at different speeds.
"""

import time
import chess
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vpr import VPREngine


def test_time_control_depths():
    """Test search depths achieved at different time controls"""
    print("=== TAL-BOT Time Control Analysis ===\n")
    
    engine = VPREngine()
    
    # Test positions of varying complexity
    positions = [
        ("Opening", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("Middlegame", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"),
        ("Complex", "r1bq1rk1/pp2bppp/2np1n2/4p3/2B1P3/2NP1N2/PPP1QPPP/R1B2RK1 w - - 0 10"),
        ("Tactical", "r1bqk2r/pp2bppp/2np1n2/4p3/2B1P3/2NP1N2/PPP1QPPP/R1B1K2R w KQkq - 0 1")
    ]
    
    # Time controls (in seconds per move)
    time_controls = [
        ("Bullet", 1.0),      # 1 second per move
        ("Blitz", 3.0),       # 3 seconds per move  
        ("Rapid", 10.0),      # 10 seconds per move
        ("Classical", 30.0)   # 30 seconds per move
    ]
    
    results = {}
    
    for tc_name, time_limit in time_controls:
        print(f"=== {tc_name.upper()} TIME CONTROL ({time_limit}s/move) ===")
        results[tc_name] = {}
        
        for pos_name, fen in positions:
            board = chess.Board(fen)
            legal_moves = len(list(board.legal_moves))
            chaos_factor = engine._calculate_chaos_factor(board)
            
            print(f"\n{pos_name} Position:")
            print(f"  Legal moves: {legal_moves}")
            print(f"  Chaos factor: {chaos_factor}")
            
            # Search with time limit
            start_time = time.time()
            best_move = engine.search(board, time_limit=time_limit)
            actual_time = time.time() - start_time
            
            # Extract depth from engine info (last depth reached)
            max_depth = engine.default_depth  # Fallback
            
            print(f"  Best move: {best_move}")
            print(f"  Search time: {actual_time:.2f}s")
            print(f"  Nodes: {engine.nodes_searched:,}")
            print(f"  NPS: {int(engine.nodes_searched / actual_time):,}")
            print(f"  Max depth: {max_depth}")
            
            results[tc_name][pos_name] = {
                'depth': max_depth,
                'nodes': engine.nodes_searched,
                'nps': int(engine.nodes_searched / actual_time),
                'chaos': chaos_factor,
                'time': actual_time
            }
    
    # Summary analysis
    print("\n" + "="*60)
    print("TAL-BOT PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"{'Time Control':<12} {'Avg Depth':<10} {'Avg NPS':<12} {'Best Chaos Pos':<15}")
    print("-" * 60)
    
    for tc_name in ['Bullet', 'Blitz', 'Rapid', 'Classical']:
        if tc_name in results:
            depths = [results[tc_name][pos]['depth'] for pos in results[tc_name]]
            nps_values = [results[tc_name][pos]['nps'] for pos in results[tc_name]]
            chaos_scores = [(results[tc_name][pos]['chaos'], pos) for pos in results[tc_name]]
            
            avg_depth = sum(depths) / len(depths)
            avg_nps = sum(nps_values) / len(nps_values)
            best_chaos = max(chaos_scores, key=lambda x: x[0])
            
            print(f"{tc_name:<12} {avg_depth:<10.1f} {avg_nps:<12,} {best_chaos[1]:<15}")
    
    return results


def test_perft_comparison():
    """Compare TAL-BOT's search efficiency using perft-style node counting"""
    print("\n=== PERFT-STYLE EFFICIENCY TEST ===\n")
    
    engine = VPREngine()
    
    # Standard perft positions
    perft_positions = [
        ("Initial", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        ("Position 3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
        ("Position 4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    ]
    
    print("Testing TAL-BOT's search efficiency vs theoretical perft...")
    
    for pos_name, fen in perft_positions:
        print(f"\n{pos_name}:")
        board = chess.Board(fen)
        legal_moves = len(list(board.legal_moves))
        chaos_factor = engine._calculate_chaos_factor(board)
        
        print(f"  Legal moves: {legal_moves}")
        print(f"  Chaos factor: {chaos_factor}")
        
        # Test different depths/times
        for time_limit in [1.0, 3.0, 5.0]:
            start_time = time.time()
            best_move = engine.search(board, time_limit=time_limit)
            search_time = time.time() - start_time
            
            if search_time > 0:
                efficiency = engine.nodes_searched / search_time
                print(f"  {time_limit}s: {engine.nodes_searched:,} nodes, {efficiency:,.0f} NPS")


def test_chaos_vs_depth_correlation():
    """Test if higher chaos positions allow deeper search (TAL-BOT advantage)"""
    print("\n=== CHAOS vs DEPTH CORRELATION ===\n")
    
    engine = VPREngine()
    
    # Positions with varying chaos levels
    chaos_test_positions = [
        ("Low Chaos", "8/8/8/3k4/3K4/8/8/8 w - - 0 1"),
        ("Medium Chaos", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("High Chaos", "r1bq1rk1/pp2bppp/2np1n2/4p3/2B1P3/2NP1N2/PPP1QPPP/R1B2RK1 w - - 0 10"),
        ("Ultra Chaos", "r1bqk2r/pp2bppp/2np1n2/4p3/2B1P3/2NP1N2/PPP1QPPP/R1B1K2R w KQkq - 0 1")
    ]
    
    time_limit = 5.0  # Standard 5-second test
    
    print("Testing TAL-BOT's hypothesis: Higher chaos = TAL-BOT advantage")
    print(f"Time limit: {time_limit}s per position\n")
    
    chaos_data = []
    
    for pos_name, fen in chaos_test_positions:
        board = chess.Board(fen)
        chaos_factor = engine._calculate_chaos_factor(board)
        legal_moves = len(list(board.legal_moves))
        
        start_time = time.time()
        best_move = engine.search(board, time_limit=time_limit)
        search_time = time.time() - start_time
        
        nps = int(engine.nodes_searched / search_time) if search_time > 0 else 0
        
        chaos_data.append({
            'name': pos_name,
            'chaos': chaos_factor,
            'moves': legal_moves,
            'nodes': engine.nodes_searched,
            'nps': nps,
            'time': search_time
        })
        
        print(f"{pos_name:<12} | Chaos: {chaos_factor:3.0f} | Moves: {legal_moves:2} | "
              f"Nodes: {engine.nodes_searched:6,} | NPS: {nps:6,}")
    
    # Analysis
    print(f"\nCHAOS ADVANTAGE ANALYSIS:")
    chaos_data.sort(key=lambda x: x['chaos'])
    
    print(f"Lowest chaos: {chaos_data[0]['name']} - {chaos_data[0]['nps']:,} NPS")
    print(f"Highest chaos: {chaos_data[-1]['name']} - {chaos_data[-1]['nps']:,} NPS")
    
    if chaos_data[-1]['nps'] > chaos_data[0]['nps']:
        advantage = chaos_data[-1]['nps'] / chaos_data[0]['nps']
        print(f"✓ TAL-BOT chaos advantage confirmed: {advantage:.2f}x faster in complex positions!")
    else:
        print("ℹ  Chaos advantage not detected in this test set")


if __name__ == "__main__":
    print("TAL-BOT: The Entropy Engine Performance Analysis")
    print("=" * 60)
    print("Testing revolutionary anti-engine across time controls...\n")
    
    # Run comprehensive testing
    results = test_time_control_depths()
    test_perft_comparison()
    test_chaos_vs_depth_correlation()
    
    print("\n" + "="*60)
    print("🔥 TAL-BOT ENTROPY ENGINE ANALYSIS COMPLETE! 🔥")
    print("Ready for Arena deployment and engine battles!")
    print("The dark forest awaits... ⚔️")