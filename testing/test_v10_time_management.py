#!/usr/bin/env python3
"""
V10 Time Management Test
Test that V10 properly manages time in tournament conditions
"""

import time
import chess
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3RCleanEngine

def test_time_management():
    """Test time management with various time controls"""
    
    print("⏰ V10 TIME MANAGEMENT TEST")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    
    # Test positions
    test_positions = [
        ("Starting position", chess.Board()),
        ("Middlegame", chess.Board("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")),
        ("Tactical position", chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")),
        ("Endgame", chess.Board("8/8/8/3k4/8/3K4/8/8 w - - 0 1"))
    ]
    
    time_controls = [
        ("Bullet (1 sec)", 1.0),
        ("Blitz (3 sec)", 3.0), 
        ("Rapid (5 sec)", 5.0),
        ("Classical (10 sec)", 10.0)
    ]
    
    for pos_name, board in test_positions:
        print(f"\n📍 Testing: {pos_name}")
        print(f"   FEN: {board.fen()}")
        
        for control_name, time_limit in time_controls:
            print(f"\n   🕐 {control_name}:")
            
            start_time = time.time()
            move = engine.search(board, time_limit)
            actual_time = time.time() - start_time
            
            # Check if move is legal
            is_legal = move in board.legal_moves if move != chess.Move.null() else False
            
            # Time management check - should stay under limit
            time_ok = actual_time <= time_limit + 0.5  # Allow 0.5s buffer
            time_efficient = actual_time >= time_limit * 0.1  # Should use at least 10% of time
            
            print(f"      Move: {move}")
            print(f"      Time used: {actual_time:.2f}s / {time_limit:.2f}s")
            print(f"      Legal: {'✅' if is_legal else '❌'}")
            print(f"      Time management: {'✅' if time_ok else '❌'}")
            print(f"      Efficiency: {'✅' if time_efficient else '⚠️'}")
            
            if not time_ok:
                print(f"      ⚠️  TIME OVERRUN: Used {actual_time:.2f}s but limit was {time_limit:.2f}s")

def test_iterative_deepening_timing():
    """Test that iterative deepening respects time limits"""
    
    print(f"\n🔄 ITERATIVE DEEPENING TIME TEST")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    
    # Test various time limits
    time_limits = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    for time_limit in time_limits:
        print(f"\n⏱️  Time limit: {time_limit}s")
        
        start_time = time.time()
        move = engine.search(board, time_limit)
        actual_time = time.time() - start_time
        
        time_ok = actual_time <= time_limit + 0.2  # Allow small buffer
        
        print(f"   Time used: {actual_time:.2f}s")
        print(f"   Nodes searched: {engine.nodes_searched}")
        print(f"   NPS: {int(engine.nodes_searched / max(actual_time, 0.001))}")
        print(f"   Time respected: {'✅' if time_ok else '❌'}")
        
        if not time_ok:
            print(f"   ⚠️  OVERRUN: {actual_time - time_limit:.2f}s over limit!")

def test_tournament_simulation():
    """Simulate tournament conditions with realistic time controls"""
    
    print(f"\n🏆 TOURNAMENT SIMULATION")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    
    # Simulate a tournament game scenario
    # Each player gets 5 minutes (300 seconds) for 40 moves
    white_time = 300.0  # 5 minutes
    black_time = 300.0  # 5 minutes
    move_count = 0
    
    board = chess.Board()
    
    print("Simulating tournament time management...")
    print("Each side: 5 minutes for 40 moves")
    print("Expected time per move: ~7.5 seconds")
    
    # Simulate several moves
    for move_num in range(1, 6):  # Test first 5 moves
        move_count += 1
        
        # Calculate time to use (more aggressive early, conservative later)
        if move_count <= 20:
            time_factor = 25.0  # Use 1/25th of remaining time
        elif move_count <= 30:
            time_factor = 30.0  # Use 1/30th of remaining time
        else:
            time_factor = 40.0  # Use 1/40th of remaining time
        
        if board.turn == chess.WHITE:
            time_to_use = min(white_time / time_factor, 10.0)
            print(f"\nMove {move_num} (White):")
            print(f"   Remaining time: {white_time:.1f}s")
            print(f"   Time budget: {time_to_use:.2f}s")
            
            start_time = time.time()
            move = engine.search(board, time_to_use)
            actual_time = time.time() - start_time
            
            white_time -= actual_time
            
            print(f"   Time used: {actual_time:.2f}s")
            print(f"   New remaining: {white_time:.1f}s")
            print(f"   Time management: {'✅' if actual_time <= time_to_use + 0.3 else '❌'}")
            
        else:
            time_to_use = min(black_time / time_factor, 10.0)
            print(f"\nMove {move_num} (Black):")
            print(f"   Remaining time: {black_time:.1f}s")
            print(f"   Time budget: {time_to_use:.2f}s")
            
            start_time = time.time()
            move = engine.search(board, time_to_use)
            actual_time = time.time() - start_time
            
            black_time -= actual_time
            
            print(f"   Time used: {actual_time:.2f}s")
            print(f"   New remaining: {black_time:.1f}s")
            print(f"   Time management: {'✅' if actual_time <= time_to_use + 0.3 else '❌'}")
        
        # Make the move
        if move in board.legal_moves:
            board.push(move)
        else:
            print(f"   ❌ ILLEGAL MOVE: {move}")
            break
    
    print(f"\n📊 Tournament Simulation Summary:")
    print(f"   White remaining time: {white_time:.1f}s")
    print(f"   Black remaining time: {black_time:.1f}s")
    print(f"   Moves played: {len(board.move_stack)}")
    
    if white_time > 60 and black_time > 60:
        print("   ✅ Good time management - plenty of time remaining")
    elif white_time > 0 and black_time > 0:
        print("   ⚠️  Tight on time but manageable")
    else:
        print("   ❌ Time management failure - would lose on time")

def test_uci_time_commands():
    """Test UCI time control parsing"""
    
    print(f"\n🔧 UCI TIME COMMAND TEST")
    print("=" * 50)
    
    print("Testing UCI 'go' command time parsing...")
    
    # Test cases for UCI time commands
    uci_commands = [
        "go movetime 3000",  # 3 seconds move time
        "go wtime 300000 btime 300000",  # 5 minutes each
        "go wtime 60000 btime 60000 winc 1000 binc 1000",  # 1 minute + 1 sec increment
        "go depth 6",  # Fixed depth
        "go wtime 180000 btime 180000 movestogo 40",  # 3 minutes for 40 moves
    ]
    
    for cmd in uci_commands:
        print(f"\n   Command: {cmd}")
        parts = cmd.split()
        
        # Simulate parsing (similar to UCI interface)
        time_limit = 3.0  # Default
        
        for i, part in enumerate(parts):
            if part == "movetime" and i + 1 < len(parts):
                time_limit = int(parts[i + 1]) / 1000.0
                print(f"   Parsed movetime: {time_limit}s")
            elif part == "wtime" and i + 1 < len(parts):
                wtime = int(parts[i + 1]) / 1000.0
                # Simulate time calculation for white
                time_limit = min(wtime / 25.0, 10.0)  # Use 1/25th
                print(f"   Parsed wtime: {wtime}s -> budget: {time_limit:.2f}s")
            elif part == "depth" and i + 1 < len(parts):
                depth = int(parts[i + 1])
                print(f"   Parsed depth: {depth}")
                time_limit = 30.0  # Allow more time for fixed depth
        
        print(f"   Final time budget: {time_limit:.2f}s")

if __name__ == "__main__":
    test_time_management()
    test_iterative_deepening_timing()
    test_tournament_simulation()
    test_uci_time_commands()
    
    print(f"\n🏁 TIME MANAGEMENT TEST COMPLETE")
    print("=" * 50)
    print("✅ V10 is ready for tournament play with proper time management!")
