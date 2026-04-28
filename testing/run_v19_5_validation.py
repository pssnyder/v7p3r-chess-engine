#!/usr/bin/env python3
"""
V19.5.2 vs V18.4 Validation Tournament

Quick 4-game validation at 5min+4s blitz to validate ITERATIVE DEEPENING TIMEOUT FIX:
- Added timeout checks after each _recursive_search call in iterative deepening
- Prevents loop from continuing after timeout (was wasting time in loop overhead)
- Fixes 135-180s moves when given 30s time limit

SUCCESS CRITERIA:
✓ 0 timeouts (PRIMARY - timeout fix validation)
✓ Win rate ≥45% (maintains strength)
✓ 0 crashes (stability)
"""

import sys
import os
import chess
import chess.engine
import time
from pathlib import Path
from typing import Dict, List

# Add src to path for direct engine import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "lichess" / "engines" / "V7P3R_v18.4_20260417" / "src"))

class SimpleGame:
    """Run a single game between two engines"""
    
    def __init__(self, engine1_path: str, engine2_path: str, time_control: Dict):
        self.engine1_path = engine1_path
        self.engine2_path = engine2_path
        self.time_control = time_control
        
    def run(self, engine1_white: bool = True) -> Dict:
        """Run game and return result"""
        from v7p3r import V7P3REngine
        
        # Load engines with separate imports
        if engine1_white:
            white_name = "v19.5.1"
            black_name = "v18.4"
        else:
            white_name = "v18.4"
            black_name = "v19.5.1"
        
        board = chess.Board()
        moves = []
        
        # Time tracking
        white_time = self.time_control['base']
        black_time = self.time_control['base']
        increment = self.time_control['increment']
        
        # Load engines
        print(f"  White: {white_name}, Black: {black_name}")
        
        # Import engines from different paths
        # v19.5.1 from src/, v18.4 from lichess/engines/
        if engine1_white:
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from v7p3r import V7P3REngine as Engine1
            sys.path.pop(0)
            
            sys.path.insert(0, str(Path(__file__).parent.parent / "lichess" / "engines" / "V7P3R_v18.4_20260417" / "src"))
            from v7p3r import V7P3REngine as Engine2
            sys.path.pop(0)
            
            white_engine = Engine1()
            black_engine = Engine2()
        else:
            sys.path.insert(0, str(Path(__file__).parent.parent / "lichess" / "engines" / "V7P3R_v18.4_20260417" / "src"))
            from v7p3r import V7P3REngine as Engine1
            sys.path.pop(0)
            
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from v7p3r import V7P3REngine as Engine2
            sys.path.pop(0)
            
            white_engine = Engine1()
            black_engine = Engine2()
        
        move_num = 1
        result = None
        termination = None
        
        while not board.is_game_over() and move_num <= 200:  # Max 200 moves
            # Determine current player
            if board.turn == chess.WHITE:
                engine = white_engine
                time_limit = white_time
            else:
                engine = black_engine
                time_limit = black_time
            
            # Make move with time tracking
            start = time.time()
            try:
                move = engine.search(board, time_limit=min(time_limit, 30.0))  # Max 30s per move
                elapsed = time.time() - start
            except Exception as e:
                print(f"    ERROR at move {move_num}: {e}")
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                termination = "crash"
                break
            
            if move is None or move == chess.Move.null():
                print(f"    Illegal move at move {move_num}")
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                termination = "illegal"
                break
            
            # Deduct time and add increment
            if board.turn == chess.WHITE:
                white_time -= elapsed
                white_time += increment
                if white_time <= 0:
                    print(f"    White timeout at move {move_num}")
                    result = "0-1"
                    termination = "timeout"
                    break
            else:
                black_time -= elapsed
                black_time += increment
                if black_time <= 0:
                    print(f"    Black timeout at move {move_num}")
                    result = "1-0"
                    termination = "timeout"
                    break
            
            # Make move
            moves.append(move.uci())
            board.push(move)
            
            if move_num % 10 == 0:
                print(f"    Move {move_num}: {move.uci()} (W:{white_time:.1f}s, B:{black_time:.1f}s)")
            
            move_num += 1
        
        # Determine result if not already set
        if result is None:
            if board.is_checkmate():
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                termination = "checkmate"
            elif board.is_stalemate():
                result = "1/2-1/2"
                termination = "stalemate"
            elif board.is_insufficient_material():
                result = "1/2-1/2"
                termination = "insufficient"
            elif board.can_claim_draw():
                result = "1/2-1/2"
                termination = "repetition"
            else:
                result = "1/2-1/2"
                termination = "moves"
        
        # Calculate score for engine1
        if engine1_white:
            if result == "1-0":
                engine1_score = 1.0
            elif result == "0-1":
                engine1_score = 0.0
            else:
                engine1_score = 0.5
        else:
            if result == "0-1":
                engine1_score = 1.0
            elif result == "1-0":
                engine1_score = 0.0
            else:
                engine1_score = 0.5
        
        return {
            'result': result,
            'termination': termination,
            'moves': len(moves),
            'engine1_score': engine1_score,
            'white': white_name,
            'black': black_name
        }

def main():
    print("=" * 80)
    print("V19.5.2 VS V18.4 TIMEOUT FIX VALIDATION (4 GAMES)")
    print("=" * 80)
    print("Format: 4 games, 5min+4s blitz")
    print("Primary Goal: 0 timeouts (iterative deepening timeout check)")
    print("Secondary: ≥45% win rate, 0 crashes")
    print("=" * 80)
    
    # Time control: 5 minutes + 4 seconds increment
    time_control = {
        'base': 300.0,  # 5 minutes in seconds
        'increment': 4.0
    }
    
    engine1_path = "src/v7p3r.py"  # v19.5
    engine2_path = "lichess/engines/V7P3R_v18.4_20260417/src/v7p3r.py"  # v18.4
    
    game_runner = SimpleGame(engine1_path, engine2_path, time_control)
    
    results = []
    wins = 0
    losses = 0
    draws = 0
    timeouts = 0
    crashes = 0
    
    # Run 4 games for quick timeout fix validation (2 as white, 2 as black)
    for game_num in range(1, 5):
        engine1_white = (game_num % 2 == 1)  # Alternate colors
        
        print(f"\nGame {game_num}/4:")
        result = game_runner.run(engine1_white)
        results.append(result)
        
        # Update stats
        if result['engine1_score'] == 1.0:
            wins += 1
            outcome = "WIN"
        elif result['engine1_score'] == 0.0:
            losses += 1
            outcome = "LOSS"
        else:
            draws += 1
            outcome = "DRAW"
        
        if result['termination'] == 'timeout':
            timeouts += 1
        if result['termination'] == 'crash':
            crashes += 1
        
        print(f"  Result: {result['result']} ({result['termination']}) - {outcome} for v19.5.2")
        print(f"  Current score: {wins}W - {losses}L - {draws}D ({(wins + draws*0.5)/game_num*100:.1f}%)")
    
    # Final summary
    total_games = len(results)
    score = wins + draws * 0.5
    win_rate = score / total_games * 100
    
    print(f"\n{'=' * 80}")
    print("TOURNAMENT RESULTS")
    print(f"{'=' * 80}")
    print(f"Total games:  {total_games}")
    print(f"v19.5.2 score:  {score}/{total_games} ({win_rate:.1f}%)")
    print(f"  Wins:       {wins}")
    print(f"  Losses:     {losses}")
    print(f"  Draws:      {draws}")
    print(f"")
    print(f"Critical issues:")
    print(f"  Timeouts:   {timeouts}")
    print(f"  Crashes:    {crashes}")
    print(f"")
    
    # Deployment recommendation
    print(f"{'=' * 80}")
    print("DEPLOYMENT RECOMMENDATION")
    print(f"{'=' * 80}")
    
    passed = True
    
    # PRIORITY 1: Timeouts (this is what we're fixing!)
    if timeouts > 0:
        print(f"✗ {timeouts} timeout(s) detected - CRITICAL FAILURE")
        passed = False
    else:
        print(f"✓ 0 timeouts (iterative deepening FIXED)")
    
    # PRIORITY 2: Win rate
    if win_rate < 45.0:
        print(f"✗ Win rate {win_rate:.1f}% < 45% threshold")
        passed = False
    else:
        print(f"✓ Win rate {win_rate:.1f}% ≥ 45% threshold")
    
    # PRIORITY 3: Crashes
    if crashes > 0:
        print(f"✗ {crashes} crash(es) detected")
        passed = False
    else:
        print(f"✓ 0 crashes (engine stable)")
    
    if passed:
        print(f"\n🎉 PASS: v19.5.2 is ready for deployment!")
    else:
        print(f"\n⚠️  FAIL: v19.5.2 needs additional fixes before deployment")
    
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
