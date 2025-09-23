#!/usr/bin/env python3
"""
V11.5 Simple Performance Analysis
=================================

Quick analysis of key performance bottlenecks without complex imports.
"""

import time
import chess

def test_print_overhead():
    """Test UCI print statement overhead"""
    print("=== UCI OUTPUT OVERHEAD TEST ===")
    
    # Test without printing
    start_time = time.time()
    for i in range(10000):
        info_string = f"info depth {i} score cp 100 nodes {i*10} time {i} nps {i*100}"
    no_print_time = time.time() - start_time
    
    # Test with printing (simulating UCI output)
    start_time = time.time()
    for i in range(1000):  # Fewer iterations to avoid console flood
        print(f"info depth {i} score cp 100 nodes {i*10} time {i} nps {i*100}")
    print_time = time.time() - start_time
    
    print(f"10000 string formations (no print): {no_print_time:.3f}s")
    print(f"1000 print statements: {print_time:.3f}s") 
    print(f"Print overhead per line: {print_time*1000/1000:.3f}ms")
    
    # Estimate impact during search
    print(f"If engine prints 50 UCI lines per second:")
    print(f"  Overhead: {50 * print_time*1000/1000:.1f}ms/second")
    print(f"  Performance loss: {(50 * print_time*1000/1000)/1000*100:.1f}% of CPU time")

def test_move_generation_speed():
    """Test basic move generation performance"""
    print("\n=== MOVE GENERATION SPEED TEST ===")
    
    positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # After e4
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",  # Italian game
    ]
    
    for i, fen in enumerate(positions):
        board = chess.Board(fen)
        
        start_time = time.time()
        for _ in range(1000):
            moves = list(board.legal_moves)
        elapsed = time.time() - start_time
        
        print(f"Position {i+1}: {len(moves)} moves, 1000 generations in {elapsed:.3f}s ({1000/elapsed:.0f}/sec)")

def test_board_operations():
    """Test basic board operation speeds"""
    print("\n=== BOARD OPERATION SPEED TEST ===")
    
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    moves = list(board.legal_moves)[:10]  # First 10 moves
    
    # Test make/unmake speed
    start_time = time.time()
    for _ in range(1000):
        for move in moves:
            board.push(move)
            board.pop()
    elapsed = time.time() - start_time
    
    print(f"10000 make/unmake operations: {elapsed:.3f}s ({10000/elapsed:.0f}/sec)")
    
    # Test FEN generation speed
    start_time = time.time()
    for _ in range(1000):
        fen = board.fen()
    elapsed = time.time() - start_time
    
    print(f"1000 FEN generations: {elapsed:.3f}s ({1000/elapsed:.0f}/sec)")
    
    # Test piece_at speed
    start_time = time.time()
    for _ in range(1000):
        for square in range(64):
            piece = board.piece_at(square)
    elapsed = time.time() - start_time
    
    print(f"64000 piece_at calls: {elapsed:.3f}s ({64000/elapsed:.0f}/sec)")

def analyze_search_complexity():
    """Estimate search complexity"""
    print("\n=== SEARCH COMPLEXITY ANALYSIS ===")
    
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    moves = list(board.legal_moves)
    
    print(f"Position has {len(moves)} legal moves")
    print(f"At depth 3: ~{len(moves)**3:,} positions to evaluate")
    print(f"At depth 4: ~{len(moves)**4:,} positions to evaluate")
    
    # Estimate with typical branching factor
    avg_moves = 35
    print(f"\nWith average 35 moves per position:")
    print(f"Depth 3: ~{avg_moves**3:,} positions")
    print(f"Depth 4: ~{avg_moves**4:,} positions")
    print(f"Depth 5: ~{avg_moves**5:,} positions")
    
    # Performance targets
    print(f"\nFor 5-second move time:")
    print(f"At 1,000 NPS: Can search to depth ~2.5")
    print(f"At 10,000 NPS: Can search to depth ~3.5") 
    print(f"At 100,000 NPS: Can search to depth ~4.5")

def main():
    print("V11.5 SIMPLE PERFORMANCE ANALYSIS")
    print("==================================")
    
    test_print_overhead()
    test_move_generation_speed()
    test_board_operations()
    analyze_search_complexity()
    
    print("\n=== KEY FINDINGS ===")
    print("1. UCI print overhead can consume significant CPU time")
    print("2. Move generation is fast (~1000+/sec)")
    print("3. Basic board operations are efficient")
    print("4. Search complexity grows exponentially with depth")
    print("5. Need 10,000+ NPS to reach competitive search depths")

if __name__ == "__main__":
    main()