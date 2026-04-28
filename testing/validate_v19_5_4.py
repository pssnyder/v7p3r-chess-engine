#!/usr/bin/env python3
"""
V19.5.6 vs V18.4 Validation Tournament

Quick 4-6 game tournament at 5min+4s blitz to validate time management fix.

SUCCESS CRITERIA:
✓ Win rate ≥45% (maintains strength vs v18.4 baseline)
✓ 0 timeouts (time management compliance)
✓ 0 crashes (stability)

DEPLOYMENT DECISION:
- If all criteria met: Deploy v19.5.6 to production
- If timeout issues persist: Continue debugging
"""

import subprocess
import time
import chess
from pathlib import Path
from typing import Dict, Tuple, Optional

# Engine paths
V19_5_4_PATH = Path(__file__).parent.parent / "src" / "v7p3r_uci.py"
V18_4_PATH = Path(__file__).parent.parent / "lichess" / "engines" / "V7P3R_v18.4_20260417" / "src" / "v7p3r_uci.py"

class UCIEngine:
    """UCI engine wrapper via subprocess"""
    def __init__(self, path: Path):
        self.path = path
        self.process = subprocess.Popen(
            ["python", str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Initialize
        self._send("uci")
        while True:
            line = self._receive()
            if not line:
                break
            if line == "uciok":
                break
            if line.startswith("id name"):
                self.name = line.split("id name ")[1]
        
        self._send("isready")
        while self._receive() != "readyok":
            pass
    
    def _send(self, cmd: str):
        """Send command to engine"""
        try:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()
        except:
            pass
    
    def _receive(self) -> str:
        """Receive line from engine"""
        try:
            return self.process.stdout.readline().strip()
        except:
            return ""
    
    def search(self, board: chess.Board, time_limit: float) -> Tuple[Optional[chess.Move], float]:
        """Search position with time limit, return (move, elapsed_time)"""
        self._send(f"position fen {board.fen()}")
        self._send(f"go movetime {int(time_limit * 1000)}")
        
        start = time.time()
        while True:
            line = self._receive()
            if not line:
                return None, time.time() - start
            if line.startswith("bestmove"):
                try:
                    move_str = line.split()[1]
                    move = chess.Move.from_uci(move_str)
                    return move, time.time() - start
                except:
                    return None, time.time() - start
    
    def quit(self):
        """Quit engine"""
        self._send("quit")
        self.process.wait(timeout=2)

def play_game(white_engine: UCIEngine, black_engine: UCIEngine, 
              base_time: float = 300.0, increment: float = 4.0) -> Dict:
    """Play one game with time control"""
    board = chess.Board()
    white_time = base_time
    black_time = base_time
    
    move_num = 1
    timeout_occurred = False
    crash_occurred = False
    
    print(f"  Game: {white_engine.name} (W) vs {black_engine.name} (B)")
    
    while not board.is_game_over() and move_num <= 200:
        engine = white_engine if board.turn == chess.WHITE else black_engine
        time_remaining = white_time if board.turn == chess.WHITE else black_time
        
        # Use min of remaining time and 60s per move (generous limit)
        time_limit = min(time_remaining, 60.0)
        
        move, elapsed = engine.search(board, time_limit)
        
        if move is None:
            print(f"    ERROR: Crashed at move {move_num}")
            crash_occurred = True
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            break
        
        # Update time
        if board.turn == chess.WHITE:
            white_time -= elapsed
            white_time += increment
            if white_time < 0:
                print(f"    TIMEOUT: White at move {move_num} ({elapsed:.2f}s)")
                timeout_occurred = True
                result = "0-1"
                break
        else:
            black_time -= elapsed
            black_time += increment
            if black_time < 0:
                print(f"    TIMEOUT: Black at move {move_num} ({elapsed:.2f}s)")
                timeout_occurred = True
                result = "1-0"
                break
        
        board.push(move)
        
        if board.turn == chess.BLACK:
            move_num += 1
    
    if not timeout_occurred and not crash_occurred:
        result = board.result()
    
    print(f"    Result: {result} after {move_num-1} moves")
    
    return {
        "result": result,
        "moves": move_num - 1,
        "timeout": timeout_occurred,
        "timeout_color": board.turn if timeout_occurred else None,
        "crash": crash_occurred,
        "white": white_engine.name,
        "black": black_engine.name
    }

def run_tournament(num_games: int = 6):
    """Run validation tournament"""
    print(f"V19.5.4 vs V18.4 Validation Tournament")
    print(f"Time control: 5min+4s blitz")
    print(f"Games: {num_games}\n")
    
    results = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "timeouts": 0,
        "crashes": 0,
        "games": []
    }
    
    for game_num in range(1, num_games + 1):
        print(f"\nGame {game_num}/{num_games}:")
        
        # Alternate colors
        if game_num % 2 == 1:
            # v19.5.4 is white
            white = UCIEngine(V19_5_4_PATH)
            black = UCIEngine(V18_4_PATH)
            v19_is_white = True
        else:
            # v19.5.4 is black
            white = UCIEngine(V18_4_PATH)
            black = UCIEngine(V19_5_4_PATH)
            v19_is_white = False
        
        game_result = play_game(white, black, base_time=300.0, increment=4.0)
        
        white.quit()
        black.quit()
        
        # Count result from v19.5.4 perspective
        if game_result["timeout"]:
            # Only count if v19.5.4 timed out
            if v19_is_white and game_result["timeout_color"] == chess.WHITE:
                results["timeouts"] += 1
            elif not v19_is_white and game_result["timeout_color"] == chess.BLACK:
                results["timeouts"] += 1
        
        if game_result["crash"]:
            results["crashes"] += 1
        
        if v19_is_white:
            if game_result["result"] == "1-0":
                results["wins"] += 1
            elif game_result["result"] == "0-1":
                results["losses"] += 1
            else:
                results["draws"] += 1
        else:
            if game_result["result"] == "0-1":
                results["wins"] += 1
            elif game_result["result"] == "1-0":
                results["losses"] += 1
            else:
                results["draws"] += 1
        
        results["games"].append(game_result)
    
    # Final report
    print("\n" + "="*60)
    print("VALIDATION TOURNAMENT RESULTS")
    print("="*60)
    print(f"V19.5.4 vs V18.4 @ 5min+4s blitz")
    print(f"\nResults: {results['wins']}W-{results['losses']}L-{results['draws']}D")
    total_games = results['wins'] + results['losses'] + results['draws']
    score = results['wins'] + results['draws'] * 0.5
    win_rate = (score / total_games * 100) if total_games > 0 else 0
    print(f"Score: {score}/{total_games} ({win_rate:.1f}%)")
    print(f"\nTimeouts: {results['timeouts']}")
    print(f"Crashes: {results['crashes']}")
    
    # Success criteria
    print("\n" + "="*60)
    print("SUCCESS CRITERIA")
    print("="*60)
    
    criteria_met = True
    
    print(f"✓ Win rate ≥45%: {'PASS' if win_rate >= 45 else 'FAIL'} ({win_rate:.1f}%)")
    if win_rate < 45:
        criteria_met = False
    
    print(f"✓ 0 timeouts: {'PASS' if results['timeouts'] == 0 else 'FAIL'} ({results['timeouts']} timeouts)")
    if results['timeouts'] > 0:
        criteria_met = False
    
    print(f"✓ 0 crashes: {'PASS' if results['crashes'] == 0 else 'FAIL'} ({results['crashes']} crashes)")
    if results['crashes'] > 0:
        criteria_met = False
    
    print("\n" + "="*60)
    if criteria_met:
        print("✓ ALL CRITERIA MET - READY FOR DEPLOYMENT")
    else:
        print("✗ CRITERIA NOT MET - CONTINUE DEBUGGING")
    print("="*60)
    
    return criteria_met

if __name__ == "__main__":
    import sys
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    success = run_tournament(num_games)
    sys.exit(0 if success else 1)
