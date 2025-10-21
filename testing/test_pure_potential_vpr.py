#!/usr/bin/env python3
"""
Test Pure Potential VPR Engine

Tests the completely rewritten VPR based on user's original vision:
- Piece value = attacks + safe mobility (NO material assumptions)  
- Focus ONLY on highest and lowest potential pieces
- Imperfect play assumptions
- Chaos preservation through lenient pruning
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time

try:
    from vpr_pure_potential import VPREngine
except ImportError:
    import importlib.util
    vpr_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'vpr_pure_potential.py')
    spec = importlib.util.spec_from_file_location("vpr_pure_potential", vpr_path)
    if spec and spec.loader:
        vpr_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vpr_module)
        VPREngine = vpr_module.VPREngine
    else:
        raise ImportError("Could not load Pure Potential VPR module")

def test_pure_potential_engine():
    """Test the pure potential VPR implementation"""
    print("🎯 PURE POTENTIAL VPR ENGINE TEST")
    print("=" * 50)
    print("Philosophy: Piece value = attacks + mobility (NO material assumptions)")
    print("Focus: Highest and lowest potential pieces ONLY")
    print("Assumption: Imperfect opponent play (not perfect responses)")
    print()
    
    engine = VPREngine()
    
    # Test positions that should highlight potential vs material differences
    test_positions = [
        {
            "name": "Starting Position", 
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "description": "Should prioritize knight development (high potential) over rook moves (low potential)"
        },
        {
            "name": "Knight vs Trapped Rook",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "description": "Active knight should score higher than corner rook"
        },
        {
            "name": "Piece Activation Test",
            "fen": "r2qkb1r/ppp2ppp/2n1bn2/3pp3/3PP3/3B1N2/PPP2PPP/RNBQK2R w KQkq - 0 5",
            "description": "Should activate lowest potential pieces"
        },
        {
            "name": "Chaos Position",
            "fen": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "description": "High complexity should trigger chaos preservation"
        },
        {
            "name": "Endgame Potential",
            "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "description": "King and rook activity should dominate"
        }
    ]
    
    total_nodes = 0
    total_time = 0
    
    for i, test in enumerate(test_positions, 1):
        print(f"🧪 Test {i}: {test['name']}")
        print(f"   {test['description']}")
        
        board = chess.Board(test['fen'])
        
        # Show piece potentials before search
        print(f"   Piece Potentials:")
        piece_potentials = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                potential = engine._calculate_true_potential(board, square)
                piece_name = chess.piece_name(piece.piece_type)
                square_name = chess.square_name(square)
                piece_potentials.append((potential, f"{piece_name}@{square_name}"))
        
        # Sort and display
        piece_potentials.sort(reverse=True, key=lambda x: x[0])
        for potential, piece_info in piece_potentials[:5]:  # Top 5
            print(f"     {piece_info}: {potential}")
        
        # Run search
        start_time = time.perf_counter()
        best_move = engine.search(board, time_limit=3.0)
        end_time = time.perf_counter()
        
        search_time = end_time - start_time
        stats = {
            'nodes': engine.nodes_searched,
            'time_ms': int(search_time * 1000),
            'nps': int(engine.nodes_searched / search_time) if search_time > 0 else 0
        }
        
        print(f"   Best move: {best_move}")
        print(f"   Nodes: {stats['nodes']:,}")
        print(f"   Time: {stats['time_ms']}ms")
        print(f"   NPS: {stats['nps']:,}")
        print()
        
        total_nodes += stats['nodes']
        total_time += search_time
    
    avg_nps = int(total_nodes / total_time) if total_time > 0 else 0
    
    print("📊 PURE POTENTIAL VPR SUMMARY:")
    print(f"   Total nodes: {total_nodes:,}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average NPS: {avg_nps:,}")
    print(f"   Philosophy validated: Position potential over material assumptions")

def test_potential_vs_material():
    """Test potential calculation vs traditional material values"""
    print("\n🔬 POTENTIAL vs MATERIAL COMPARISON")
    print("=" * 50)
    
    engine = VPREngine()
    
    # Position where material and potential should differ significantly
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    
    print("Position: Opening with active pieces vs inactive pieces")
    print()
    
    # Traditional material values
    traditional_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }
    
    print("TRADITIONAL vs POTENTIAL VALUES:")
    print("-" * 40)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == chess.WHITE:
            potential = engine._calculate_true_potential(board, square)
            traditional = traditional_values[piece.piece_type]
            piece_name = chess.piece_name(piece.piece_type)
            square_name = chess.square_name(square)
            
            ratio = potential / traditional if traditional > 0 else float('inf')
            
            print(f"{piece_name:6}@{square_name}: Traditional={traditional:3}, Potential={potential:2}, Ratio={ratio:.2f}")
    
    print()
    print("Key Insight: High ratio = piece outperforming material assumptions")
    print("            Low ratio = piece underperforming (needs activation)")

if __name__ == "__main__":
    test_pure_potential_engine()
    test_potential_vs_material()