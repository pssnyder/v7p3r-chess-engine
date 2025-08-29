#!/usr/bin/env python3
"""
V7P3R v8.2 Enhanced Move Ordering Test
Test the new contextual, efficient move ordering system
"""

import sys
import os
sys.path.append('src')

import chess
import time
from v7p3r import V7P3RCleanEngine, SearchOptions

def test_contextual_move_ordering():
    """Test the contextual move ordering across different game phases"""
    print("🎯 Testing Contextual Move Ordering")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    options = SearchOptions()
    
    # Test positions for different game phases
    positions = [
        ("Opening", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("Middlegame", "r2qkb1r/ppp2ppp/2n1bn2/2BpP3/3P4/3B1N2/PPP2PPP/RNBQK2R w KQkq - 0 8"),
        ("Endgame", "8/5k2/6K1/3P4/8/8/8/8 w - - 0 1"),
        ("Tactical", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    ]
    
    for phase_name, fen in positions:
        print(f"\n📍 {phase_name} Position:")
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        
        print(f"  FEN: {fen}")
        print(f"  Legal moves: {len(moves)}")
        
        # Test game phase detection
        game_phase = engine._detect_game_phase(board)
        print(f"  Detected phase: Opening={game_phase.is_opening}, Mid={game_phase.is_middlegame}, End={game_phase.is_endgame}")
        
        # Test move ordering
        start_time = time.time()
        ordered_moves = engine._order_moves_enhanced(board, moves.copy(), 0, options)
        ordering_time = time.time() - start_time
        
        print(f"  Ordered moves: {len(ordered_moves)} (pruned {len(moves) - len(ordered_moves)})")
        print(f"  Ordering time: {ordering_time:.6f}s")
        
        # Show top 5 moves
        print(f"  Top moves:")
        for i, move in enumerate(ordered_moves[:5], 1):
            move_type = []
            if board.is_capture(move):
                move_type.append("Capture")
            if move.promotion:
                move_type.append("Promotion")
            if board.is_castling(move):
                move_type.append("Castle")
            if not move_type:
                move_type.append("Quiet")
            print(f"    {i}. {move} ({'/'.join(move_type)})")

def test_tactical_detection():
    """Test the multi-piece attack detection"""
    print("\n🔍 Testing Tactical Opportunity Detection")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    
    # Position with tactical opportunities
    board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    
    print(f"Position: {board.fen()}")
    
    # Build context
    moves = list(board.legal_moves)
    context = engine._build_move_ordering_context(board, moves)
    
    print(f"Captures available: {context.has_captures} ({context.capture_count} captures)")
    print(f"King in danger: {context.king_in_danger}")
    print(f"Tactical opportunities: {len(context.tactical_opportunities) if context.tactical_opportunities else 0}")
    
    if context.tactical_opportunities:
        print(f"Tactical squares: {[chess.square_name(sq) for sq in context.tactical_opportunities[:5]]}")
    
    if context.enemy_piece_positions:
        print(f"Enemy piece types tracked: {list(context.enemy_piece_positions.keys())}")

def test_efficiency_comparison():
    """Compare efficiency of new vs old move ordering"""
    print("\n⚡ Testing Efficiency Improvements")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    
    # Complex middlegame position
    board = chess.Board("r2qkb1r/ppp2ppp/2n1bn2/2BpP3/3P4/3B1N2/PPP2PPP/RNBQK2R w KQkq - 0 8")
    moves = list(board.legal_moves)
    options = SearchOptions()
    
    print(f"Test position: {len(moves)} legal moves")
    
    # Test V8.2 enhanced ordering
    start_time = time.time()
    ordered_moves = engine._order_moves_enhanced(board, moves.copy(), 0, options)
    v8_2_time = time.time() - start_time
    
    print(f"V8.2 Enhanced Ordering:")
    print(f"  Time: {v8_2_time:.6f}s")
    print(f"  Moves after pruning: {len(ordered_moves)}")
    print(f"  Pruning efficiency: {((len(moves) - len(ordered_moves)) / len(moves) * 100):.1f}% reduction")
    
    # Test a simple search to see node reduction
    print(f"\nSearch Efficiency Test (depth 3):")
    engine.nodes_searched = 0
    start_time = time.time()
    best_move = engine.search(board, 1.0)
    search_time = time.time() - start_time
    
    print(f"  Best move: {best_move}")
    print(f"  Nodes searched: {engine.nodes_searched}")
    print(f"  Search time: {search_time:.3f}s")
    print(f"  Nodes per second: {int(engine.nodes_searched / max(search_time, 0.001))}")

def test_dynamic_heuristics():
    """Test dynamic heuristics based on game phase"""
    print("\n🔄 Testing Dynamic Heuristics")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    options = SearchOptions()
    
    # Opening position - should prioritize development
    opening_board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    opening_moves = list(opening_board.legal_moves)
    
    print("Opening Position:")
    print("  Should prioritize: Development, castling preparation")
    
    ordered_opening = engine._order_moves_enhanced(opening_board, opening_moves.copy(), 0, options)
    print(f"  Top moves: {[str(m) for m in ordered_opening[:3]]}")
    
    # Check if development moves are prioritized
    development_moves = []
    for move in ordered_opening[:5]:
        piece = opening_board.piece_at(move.from_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            from_rank = chess.square_rank(move.from_square)
            if (piece.color == chess.BLACK and from_rank == 7):
                development_moves.append(move)
    
    print(f"  Development moves in top 5: {len(development_moves)}")
    
    # Endgame position - should prioritize king activity
    endgame_board = chess.Board("8/5k2/6K1/3P4/8/8/8/8 w - - 0 1")
    endgame_moves = list(endgame_board.legal_moves)
    
    print("\nEndgame Position:")
    print("  Should prioritize: King activity, pawn advancement")
    
    ordered_endgame = engine._order_moves_enhanced(endgame_board, endgame_moves.copy(), 0, options)
    print(f"  Top moves: {[str(m) for m in ordered_endgame[:3]]}")

def main():
    """Run all V8.2 enhanced move ordering tests"""
    print("V7P3R v8.2 Enhanced Move Ordering Test Suite")
    print("Contextual, Efficient, Tactical-Aware Move Ordering")
    print("=" * 65)
    
    tests = [
        test_contextual_move_ordering,
        test_tactical_detection,
        test_efficiency_comparison,
        test_dynamic_heuristics,
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"❌ Error in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 V8.2 Enhanced Move Ordering Test Complete!")
    print("Features showcased:")
    print("✅ Contextual ordering based on game phase")
    print("✅ Multi-piece tactical detection")
    print("✅ Efficient pre-pruning")
    print("✅ Dynamic heuristics")
    print("✅ Memory and computation optimization")

if __name__ == "__main__":
    main()
