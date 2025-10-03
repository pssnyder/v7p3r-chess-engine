#!/usr/bin/env python3
"""
V7P3R v12.4 Executable Castling Test
====================================
Tests the actual V12.4.exe to verify enhanced castling behavior
"""

import chess
import chess.engine
import time
import os

def test_v12_4_castling():
    """Test V12.4 executable's castling behavior"""
    
    print("=" * 60)
    print("V7P3R v12.4 EXECUTABLE CASTLING TEST")
    print("=" * 60)
    print("Testing the actual V12.4.exe enhanced castling system")
    print()
    
    # Path to V12.4 executable
    engine_path = "dist/V7P3R_v12.4_fixed.exe"
    
    if not os.path.exists(engine_path):
        print(f"❌ ERROR: {engine_path} not found!")
        print("Please make sure V12.4 is built first.")
        return
    
    print(f"Using engine: {engine_path}")
    print()
    
    # Test positions where we want to see good castling behavior
    test_cases = [
        {
            "name": "Position 1: Early Castling Opportunity",
            "fen": "r1bqk1nr/p2ppppp/1p6/8/8/8/PPPP1PPP/RNBQK1NR w KQkq - 0 6",
            "description": "V12.4 should prefer castling or strong development over Kf1"
        },
        {
            "name": "Position 2: King Safety Priority",
            "fen": "rnbqkb1r/2pppp1p/p5pn/8/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 0 5",
            "description": "V12.4 should prioritize king safety"
        },
        {
            "name": "Position 3: Development vs Manual King Move",
            "fen": "rnbqkb1r/pppp1ppp/4pn2/8/2B1P3/2N5/PPPP1PPP/R1BQK1NR w KQkq - 1 4",
            "description": "V12.4 should avoid manual king moves"
        }
    ]
    
    results = []
    
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        print(f"✅ Successfully connected to {engine_path}")
        print(f"Engine info: {engine.id}")
        print()
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"=== Test {i}: {test_case['name']} ===")
            print(f"Description: {test_case['description']}")
            print(f"FEN: {test_case['fen']}")
            
            # Set up position
            board = chess.Board(test_case['fen'])
            print("Position:")
            print(board)
            print()
            
            # Get V12.4's move
            try:
                start_time = time.time()
                result = engine.play(board, chess.engine.Limit(time=3.0, depth=4))
                search_time = time.time() - start_time
                
                best_move = result.move
                print(f"V12.4 chose: {best_move}")
                print(f"Search time: {search_time:.1f}s")
                
                # Analyze the move
                move_analysis = analyze_move(board, best_move)
                print(f"Move type: {move_analysis['type']}")
                print(f"Assessment: {move_analysis['assessment']}")
                
                results.append({
                    'test': test_case['name'],
                    'move': str(best_move),
                    'type': move_analysis['type'],
                    'assessment': move_analysis['assessment'],
                    'search_time': search_time
                })
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                results.append({
                    'test': test_case['name'],
                    'move': 'ERROR',
                    'type': 'ERROR',
                    'assessment': 'FAILED',
                    'search_time': 0
                })
            
            print()
    
    # Summary
    print("=" * 60)
    print("V12.4 CASTLING TEST SUMMARY")
    print("=" * 60)
    
    excellent_count = 0
    good_count = 0
    poor_count = 0
    
    for result in results:
        status_icon = ""
        if result['assessment'] == 'EXCELLENT':
            status_icon = "🏰"
            excellent_count += 1
        elif result['assessment'] == 'GOOD':
            status_icon = "✅"
            good_count += 1
        elif result['assessment'] == 'POOR':
            status_icon = "❌"
            poor_count += 1
        
        print(f"{status_icon} {result['test']}: {result['move']} ({result['type']})")
    
    print()
    print(f"🏰 Excellent (Castling): {excellent_count}")
    print(f"✅ Good (Development): {good_count}")
    print(f"❌ Poor (Manual King): {poor_count}")
    print(f"Total tests: {len(results)}")
    
    success_rate = ((excellent_count + good_count) / len(results)) * 100
    print(f"\n🎯 Overall Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 OUTSTANDING: V12.4 castling enhancement is working excellently!")
    elif success_rate >= 70:
        print("👍 GOOD: V12.4 shows significant improvement in castling behavior!")
    elif success_rate >= 50:
        print("📈 PROGRESS: V12.4 is better but needs more tuning")
    else:
        print("⚠️  NEEDS WORK: V12.4 castling enhancement needs more development")

def analyze_move(board, move):
    """Analyze what type of move was made"""
    
    # Check if it's castling
    if board.is_castling(move):
        return {
            'type': 'CASTLING',
            'assessment': 'EXCELLENT'
        }
    
    # Check if it's a manual king move
    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type == chess.KING:
        return {
            'type': 'MANUAL KING MOVE',
            'assessment': 'POOR'
        }
    
    # Check if it's development
    if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
        return {
            'type': 'PIECE DEVELOPMENT',
            'assessment': 'GOOD'
        }
    
    # Check if it's a queen move (often good)
    if piece and piece.piece_type == chess.QUEEN:
        return {
            'type': 'QUEEN MOVE',
            'assessment': 'GOOD'
        }
    
    # Check if it's a pawn move
    if piece and piece.piece_type == chess.PAWN:
        return {
            'type': 'PAWN MOVE',
            'assessment': 'GOOD'
        }
    
    # Other moves
    return {
        'type': 'OTHER MOVE',
        'assessment': 'GOOD'
    }

if __name__ == "__main__":
    test_v12_4_castling()