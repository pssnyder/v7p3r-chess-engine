"""
V14.9.1 Time Management Tuning Test

Evaluate time usage across:
1. Opening positions (10-15 moves) - Should move quickly
2. Middlegame quiet positions - Moderate time, but exit on decisive PV
3. Middlegame tactical/noisy positions - Maximum depth, use full time
4. Endgame positions - Calculated time based on complexity

Test across Rapid (300s), Blitz (180s), Bullet (60s) time controls
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3REngine

class TimeManagementTest:
    def __init__(self):
        self.engine = V7P3REngine()
        self.results = []
        
    def count_captures_threats(self, board):
        """Count captures and checks available"""
        captures = 0
        checks = 0
        for move in board.legal_moves:
            if board.is_capture(move):
                captures += 1
            if board.gives_check(move):
                checks += 1
        return captures, checks
    
    def is_noisy_position(self, board):
        """Determine if position is tactically complex"""
        captures, checks = self.count_captures_threats(board)
        
        # Noisy if: many captures, checks, or in check
        if board.is_check():
            return True
        if captures >= 5:
            return True
        if checks >= 3:
            return True
        
        return False
    
    def detect_game_phase(self, board):
        """Simple game phase detection"""
        move_count = len(board.move_stack)
        
        # Count material
        piece_count = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.PAWN:
                piece_count += 1
        
        if move_count < 10:
            return "opening"
        elif piece_count <= 6:
            return "endgame"
        else:
            return "middlegame"
    
    def test_position(self, fen, description, time_control, expected_behavior):
        """Test a single position"""
        print(f"\n{'='*80}")
        print(f"Testing: {description}")
        print(f"Phase: {self.detect_game_phase(chess.Board(fen))}")
        print(f"Time Control: {time_control}s")
        print(f"Expected: {expected_behavior}")
        print(f"{'='*80}")
        
        board = chess.Board(fen)
        
        # Detect position characteristics
        is_noisy = self.is_noisy_position(board)
        captures, checks = self.count_captures_threats(board)
        game_phase = self.detect_game_phase(board)
        
        print(f"Position Analysis:")
        print(f"  - Noisy: {is_noisy}")
        print(f"  - Available captures: {captures}")
        print(f"  - Available checks: {checks}")
        print(f"  - In check: {board.is_check()}")
        print(f"  - Legal moves: {board.legal_moves.count()}")
        
        # Calculate time allocation based on game phase
        if game_phase == "opening":
            allocated_time = min(3.0, time_control / 40)  # Quick in opening
        elif game_phase == "endgame":
            allocated_time = min(5.0, time_control / 20)  # Moderate in endgame
        else:  # middlegame
            if is_noisy:
                allocated_time = min(10.0, time_control / 15)  # More time for tactics
            else:
                allocated_time = min(5.0, time_control / 25)  # Moderate for quiet
        
        print(f"\nAllocated Time: {allocated_time:.2f}s")
        
        # Search
        start_time = time.time()
        move = self.engine.search(board, time_limit=allocated_time)
        elapsed = time.time() - start_time
        
        # Analyze results
        time_efficiency = (elapsed / allocated_time) * 100
        
        result = {
            'description': description,
            'fen': fen,
            'game_phase': game_phase,
            'is_noisy': is_noisy,
            'captures': captures,
            'checks': checks,
            'allocated_time': allocated_time,
            'actual_time': elapsed,
            'time_efficiency': time_efficiency,
            'move': str(move),
            'expected': expected_behavior
        }
        
        self.results.append(result)
        
        print(f"\nResults:")
        print(f"  Move played: {move}")
        print(f"  Time used: {elapsed:.2f}s / {allocated_time:.2f}s ({time_efficiency:.1f}%)")
        
        # Evaluate against expected behavior
        if game_phase == "opening":
            if elapsed > allocated_time * 1.2:
                print(f"  ⚠️  WARNING: Opening move took too long!")
        elif is_noisy:
            if elapsed < allocated_time * 0.7:
                print(f"  ⚠️  WARNING: Tactical position didn't use enough time!")
        
        return result

def main():
    print("="*80)
    print("V14.9.1 TIME MANAGEMENT TUNING TEST")
    print("="*80)
    
    tester = TimeManagementTest()
    
    # Test positions categorized by phase and complexity
    
    # ==== OPENING POSITIONS (should move quickly) ====
    print("\n" + "="*80)
    print("OPENING POSITIONS - Should move quickly")
    print("="*80)
    
    tester.test_position(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Starting position",
        180,  # Blitz
        "Move in <2s, simple development"
    )
    
    tester.test_position(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "After 1.e4",
        180,  # Blitz
        "Quick response, standard opening"
    )
    
    tester.test_position(
        "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        "After 1.e4 Nf6 (early game)",
        180,  # Blitz
        "Quick development move"
    )
    
    # ==== MIDDLEGAME QUIET (should exit on decisive PV) ====
    print("\n" + "="*80)
    print("MIDDLEGAME QUIET - Should find plan and move decisively")
    print("="*80)
    
    tester.test_position(
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "Italian Game - Quiet middlegame",
        180,  # Blitz
        "Clear plan, exit quickly on good PV"
    )
    
    tester.test_position(
        "rnbqk2r/ppp2ppp/3bpn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6",
        "Queen's Gambit Declined - Symmetric",
        300,  # Rapid
        "Strategic position, moderate time"
    )
    
    # ==== MIDDLEGAME TACTICAL/NOISY (use full time for deep search) ====
    print("\n" + "="*80)
    print("MIDDLEGAME TACTICAL - Should use maximum time for deep search")
    print("="*80)
    
    tester.test_position(
        "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
        "Italian Game - Tactical complications",
        300,  # Rapid
        "Many captures, need deep search"
    )
    
    tester.test_position(
        "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 10",
        "Double bishops facing off - Tactical",
        180,  # Blitz
        "Pin threats, use good time for tactics"
    )
    
    tester.test_position(
        "r1bq1rk1/pp3pbp/2nppnp1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ - 0 11",
        "Pawn storm coming - Complex",
        300,  # Rapid
        "Multiple threats, deep calculation needed"
    )
    
    # ==== FORCED TACTICAL (in check or must capture) ====
    print("\n" + "="*80)
    print("FORCED TACTICAL - Maximum depth in critical positions")
    print("="*80)
    
    tester.test_position(
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "Scholar's mate threat - Check available",
        180,  # Blitz
        "Forced tactics, calculate deeply"
    )
    
    tester.test_position(
        "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "Under attack from Queen",
        180,  # Blitz
        "Defensive tactics, accurate calculation"
    )
    
    # ==== ENDGAME (calculated based on complexity) ====
    print("\n" + "="*80)
    print("ENDGAME - Moderate time for precise calculation")
    print("="*80)
    
    tester.test_position(
        "8/8/4k3/8/8/3K4/8/8 w - - 0 1",
        "King and pawn endgame - Simple",
        180,  # Blitz
        "Simple position, moderate time"
    )
    
    tester.test_position(
        "8/5k2/8/5P2/5K2/8/8/8 w - - 0 1",
        "King and pawn endgame - Opposition",
        300,  # Rapid
        "Precise calculation needed"
    )
    
    # ==== BULLET TIME CONTROL TESTS ====
    print("\n" + "="*80)
    print("BULLET TIME CONTROL (60s) - All positions must move quickly")
    print("="*80)
    
    tester.test_position(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "Opening in bullet",
        60,  # Bullet
        "Very quick, <1s"
    )
    
    tester.test_position(
        "r1bq1rk1/pp3pbp/2nppnp1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ - 0 11",
        "Complex middlegame in bullet",
        60,  # Bullet
        "Quick but accurate, <2s"
    )
    
    # ==== SUMMARY ====
    print("\n" + "="*80)
    print("TIME MANAGEMENT SUMMARY")
    print("="*80)
    
    # Analyze by game phase
    opening_times = [r for r in tester.results if r['game_phase'] == 'opening']
    middlegame_times = [r for r in tester.results if r['game_phase'] == 'middlegame']
    endgame_times = [r for r in tester.results if r['game_phase'] == 'endgame']
    
    print(f"\nOPENING POSITIONS ({len(opening_times)} tests):")
    if opening_times:
        avg_time = sum(r['actual_time'] for r in opening_times) / len(opening_times)
        avg_efficiency = sum(r['time_efficiency'] for r in opening_times) / len(opening_times)
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Average efficiency: {avg_efficiency:.1f}%")
        if avg_time > 2.5:
            print(f"  ⚠️  WARNING: Opening moves taking too long!")
        else:
            print(f"  ✅ Opening moves are quick")
    
    print(f"\nMIDDLEGAME POSITIONS ({len(middlegame_times)} tests):")
    if middlegame_times:
        quiet_positions = [r for r in middlegame_times if not r['is_noisy']]
        noisy_positions = [r for r in middlegame_times if r['is_noisy']]
        
        if quiet_positions:
            avg_quiet = sum(r['actual_time'] for r in quiet_positions) / len(quiet_positions)
            print(f"  Quiet positions - Avg time: {avg_quiet:.2f}s")
        
        if noisy_positions:
            avg_noisy = sum(r['actual_time'] for r in noisy_positions) / len(noisy_positions)
            avg_noisy_eff = sum(r['time_efficiency'] for r in noisy_positions) / len(noisy_positions)
            print(f"  Tactical positions - Avg time: {avg_noisy:.2f}s ({avg_noisy_eff:.1f}% efficiency)")
            if avg_noisy_eff < 70:
                print(f"  ⚠️  WARNING: Not using enough time on tactical positions!")
            else:
                print(f"  ✅ Good time usage on tactics")
    
    print(f"\nENDGAME POSITIONS ({len(endgame_times)} tests):")
    if endgame_times:
        avg_time = sum(r['actual_time'] for r in endgame_times) / len(endgame_times)
        print(f"  Average time: {avg_time:.2f}s")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("TUNING RECOMMENDATIONS:")
    print("="*80)
    
    # Check opening speed
    if opening_times and sum(r['actual_time'] for r in opening_times) / len(opening_times) > 2.5:
        print("\n1. OPENING TIME ALLOCATION:")
        print("   - Current: Too slow")
        print("   - Recommendation: Reduce opening time to 1-2s max")
        print("   - Implementation: Add opening book or reduce default depth in opening")
    
    # Check tactical depth
    noisy_positions = [r for r in tester.results if r['is_noisy']]
    if noisy_positions:
        avg_noisy_eff = sum(r['time_efficiency'] for r in noisy_positions) / len(noisy_positions)
        if avg_noisy_eff < 70:
            print("\n2. TACTICAL POSITION TIME:")
            print(f"   - Current: {avg_noisy_eff:.1f}% efficiency (too low)")
            print("   - Recommendation: Increase time allocation for noisy positions")
            print("   - Implementation: Detect captures/checks and extend time limit")
    
    # Check for premature exits
    print("\n3. DECISIVE PV THRESHOLD:")
    print("   - Current: Uses full allocated time")
    print("   - Recommendation: Add early exit when:")
    print("     * PV stable for 2+ iterations")
    print("     * Eval difference >200cp from alternatives")
    print("     * No tactical complications (captures/checks)")
    
    print("\n" + "="*80)
    print("Test complete! Review recommendations above.")
    print("="*80)

if __name__ == "__main__":
    main()
