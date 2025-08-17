# test_simple_performance_v4_2.py
"""
Simple performance testing for V7P3R Chess Engine v4.2
Focus on key performance metrics without complex evaluation.
"""

import chess
import time
import sys
import os

def simple_perft(board, depth):
    """Simple perft implementation"""
    if depth == 0:
        return 1
    
    count = 0
    for move in board.legal_moves:
        board.push(move)
        count += simple_perft(board, depth - 1)
        board.pop()
    
    return count

def test_legal_move_generation(fen):
    """Test legal move generation speed"""
    print(f"Testing legal move generation...")
    print(f"Position: {fen}")
    
    board = chess.Board(fen)
    
    # Test depth 4 perft
    start_time = time.time()
    nodes = simple_perft(board, 4)
    perft_time = time.time() - start_time
    
    print(f"Perft(4): {nodes:,} nodes")
    print(f"Time: {perft_time:.2f}s")
    print(f"NPS: {nodes/perft_time:,.0f}")
    
    return nodes, perft_time

def test_move_generation_only(fen, iterations=10000):
    """Test just move generation speed"""
    print(f"\nTesting move generation speed ({iterations:,} iterations)...")
    
    board = chess.Board(fen)
    
    start_time = time.time()
    total_moves = 0
    
    for _ in range(iterations):
        moves = list(board.legal_moves)
        total_moves += len(moves)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Generated {total_moves:,} moves in {elapsed:.2f}s")
    print(f"Moves per second: {total_moves/elapsed:,.0f}")
    print(f"Average moves per position: {total_moves/iterations:.1f}")

def analyze_position_complexity(fen):
    """Analyze the complexity of the test position"""
    print(f"\nPosition Analysis:")
    print(f"FEN: {fen}")
    
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    
    print(f"Legal moves: {len(legal_moves)}")
    
    # Count move types
    captures = sum(1 for move in legal_moves if board.is_capture(move))
    checks = 0
    for move in legal_moves:
        board.push(move)
        if board.is_check():
            checks += 1
        board.pop()
    
    print(f"Captures: {captures}")
    print(f"Checks: {checks}")
    print(f"Quiet moves: {len(legal_moves) - captures}")
    
    # Material count
    piece_count = 0
    for square in chess.SQUARES:
        if board.piece_at(square):
            piece_count += 1
    
    print(f"Pieces on board: {piece_count}")
    print(f"In check: {board.is_check()}")
    print(f"Can castle: {board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE) or board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK)}")

def test_expected_performance():
    """Test expected performance targets"""
    print(f"\n" + "="*50)
    print("PERFORMANCE TARGETS FOR v4.2")
    print("="*50)
    
    # Performance targets for games under 300 seconds (5 minutes)
    # At depth 4-5, we should be able to search in under 5 seconds per move
    
    target_depth = 4
    target_time = 5.0  # seconds
    target_nps = 50000  # nodes per second (conservative target)
    
    print(f"Target search depth: {target_depth}")
    print(f"Target time per move: {target_time}s")
    print(f"Target NPS: {target_nps:,}")
    print(f"Maximum nodes in target time: {target_nps * target_time:,}")
    
    # Expected search tree sizes
    print(f"\nExpected search tree pruning:")
    print(f"  Raw perft(4): ~4.6M nodes")
    print(f"  With good alpha-beta: ~460K nodes (90% reduction)")
    print(f"  With move ordering: ~46K nodes (99% reduction)")
    print(f"  Target for fast play: <23K nodes (99.5% reduction)")

def main():
    # The test position - complex middlegame
    test_fen = "r3k2r/p1ppqpb1/Bn2pnp1/3PN3/1p2P3/2N2Q2/PPPB1PpP/R3K2R w KQkq - 0 1"
    
    print("V7P3R Chess Engine Simple Performance Test v4.2")
    print("=" * 55)
    
    # Analyze the position
    analyze_position_complexity(test_fen)
    
    # Test basic move generation
    test_move_generation_only(test_fen)
    
    # Test perft performance
    nodes, perft_time = test_legal_move_generation(test_fen)
    
    # Show performance targets
    test_expected_performance()
    
    # Calculate what we need to achieve
    print(f"\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    
    current_nps = nodes / perft_time if perft_time > 0 else 0
    print(f"Current perft NPS: {current_nps:,.0f}")
    
    # For 5-second moves at depth 4, we need to reduce 4.6M nodes to something manageable
    max_nodes_for_5s = current_nps * 5
    required_reduction = (nodes - max_nodes_for_5s) / nodes * 100 if nodes > max_nodes_for_5s else 0
    
    print(f"Nodes we can search in 5s: {max_nodes_for_5s:,.0f}")
    if required_reduction > 0:
        print(f"Required search reduction: {required_reduction:.1f}%")
    else:
        print(f"Current speed is sufficient for 5s moves!")

if __name__ == "__main__":
    main()
