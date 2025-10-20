#!/usr/bin/env python3
"""
VPR Tactical Test Suite - Test VPR on specific tactical positions
Measures VPR's performance on tactical puzzles vs positional positions
"""

import sys
import os
import chess
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vpr import VPREngine

def test_tactical_suite():
    """Test VPR on various tactical positions"""
    
    # Tactical test positions with expected characteristics
    positions = [
        {
            'name': 'Fork Tactic',
            'fen': '6k1/5ppp/8/8/8/8/2N2PPP/6K1 w - - 0 1',
            'description': 'Knight fork opportunity',
            'type': 'tactical'
        },
        {
            'name': 'Pin Tactic', 
            'fen': 'r3k2r/8/8/8/8/8/4Q3/4K3 w - - 0 1',
            'description': 'Queen pin on king and rook',
            'type': 'tactical'
        },
        {
            'name': 'Back Rank Mate',
            'fen': '6k1/5ppp/8/8/8/8/8/4QK2 w - - 0 1',
            'description': 'Back rank checkmate threat',
            'type': 'tactical'
        },
        {
            'name': 'Pawn Promotion',
            'fen': '8/1P6/8/8/8/8/1k6/1K6 w - - 0 1',
            'description': 'Pawn about to promote',
            'type': 'tactical'
        },
        {
            'name': 'Opening Development',
            'fen': 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
            'description': 'Standard opening position after e4',
            'type': 'positional'
        },
        {
            'name': 'Closed Position',
            'fen': 'rnbqkbnr/ppp2ppp/4p3/3p4/3P4/4P3/PPP2PPP/RNBQKBNR w KQkq - 0 3',
            'description': 'Closed pawn structure',
            'type': 'positional'
        },
        {
            'name': 'Endgame Technique',
            'fen': '8/8/8/4k3/4K3/8/4R3/8 w - - 0 1',
            'description': 'Rook endgame technique',
            'type': 'endgame'
        }
    ]
    
    print("VPR Tactical Test Suite")
    print("=" * 50)
    
    engine = VPREngine()
    results = []
    
    for i, pos in enumerate(positions, 1):
        print(f"\n[Test {i}/{len(positions)}] {pos['name']} ({pos['type'].upper()})")
        print(f"FEN: {pos['fen']}")
        print(f"Description: {pos['description']}")
        print("-" * 40)
        
        board = chess.Board(pos['fen'])
        
        # Test with 3-second search
        start_time = time.time()
        best_move = engine.search(board, time_limit=3.0)
        search_time = time.time() - start_time
        
        nps = int(engine.nodes_searched / search_time) if search_time > 0 else 0
        
        result = {
            'name': pos['name'],
            'type': pos['type'],
            'move': best_move,
            'nodes': engine.nodes_searched,
            'time': search_time,
            'nps': nps
        }
        
        results.append(result)
        
        print(f"Best move: {best_move}")
        print(f"Nodes: {engine.nodes_searched:,}")
        print(f"Time: {search_time:.3f}s")
        print(f"NPS: {nps:,}")
        
        # Reset for next test
        engine.new_game()
    
    # Analysis by position type
    print(f"\n{'='*50}")
    print("ANALYSIS BY POSITION TYPE")
    print(f"{'='*50}")
    
    types = ['tactical', 'positional', 'endgame']
    
    for pos_type in types:
        type_results = [r for r in results if r['type'] == pos_type]
        if not type_results:
            continue
            
        avg_nodes = sum(r['nodes'] for r in type_results) / len(type_results)
        avg_nps = sum(r['nps'] for r in type_results) / len(type_results)
        avg_time = sum(r['time'] for r in type_results) / len(type_results)
        
        print(f"\n{pos_type.upper()} Positions ({len(type_results)} tests):")
        print(f"  Average nodes: {avg_nodes:,.0f}")
        print(f"  Average NPS: {avg_nps:,.0f}")
        print(f"  Average time: {avg_time:.3f}s")
        
        for r in type_results:
            print(f"    {r['name']}: {r['move']} ({r['nodes']:,} nodes)")
    
    # Performance summary
    total_nodes = sum(r['nodes'] for r in results)
    total_time = sum(r['time'] for r in results)
    overall_nps = total_nodes / total_time if total_time > 0 else 0
    
    print(f"\n{'='*50}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*50}")
    print(f"Total tests: {len(results)}")
    print(f"Total nodes: {total_nodes:,}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Overall NPS: {overall_nps:,.0f}")
    
    # Identify fastest/slowest searches
    fastest = min(results, key=lambda x: x['time'])
    slowest = max(results, key=lambda x: x['time'])
    most_nodes = max(results, key=lambda x: x['nodes'])
    
    print(f"\nFastest search: {fastest['name']} ({fastest['time']:.3f}s)")
    print(f"Slowest search: {slowest['name']} ({slowest['time']:.3f}s)")
    print(f"Most nodes: {most_nodes['name']} ({most_nodes['nodes']:,} nodes)")
    
    return results

if __name__ == "__main__":
    results = test_tactical_suite()
    
    print(f"\n🎯 VPR Tactical Testing Complete!")
    print(f"Results show VPR's performance across different position types.")
    print(f"Use this data to understand where VPR excels vs V7P3R.")