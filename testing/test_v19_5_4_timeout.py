#!/usr/bin/env python3
"""
Test v19.5.4 timeout compliance using subprocess (proper UCI)
This avoids Python module caching issues
"""

import subprocess
import time
import chess
from pathlib import Path

ENGINE_PATH = Path(__file__).parent.parent / "src" / "v7p3r_uci.py"

class UCIEngine:
    """Simple UCI engine wrapper"""
    def __init__(self):
        self.process = subprocess.Popen(
            ["python", str(ENGINE_PATH)],
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
            if line == "uciok":
                break
            if line.startswith("id name"):
                self.name = line.split("id name ")[1]
        
        self._send("isready")
        while self._receive() != "readyok":
            pass
    
    def _send(self, cmd):
        """Send command to engine"""
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()
    
    def _receive(self):
        """Receive line from engine"""
        return self.process.stdout.readline().strip()
    
    def search(self, board, time_limit):
        """Search position with time limit"""
        self._send(f"position fen {board.fen()}")
        self._send(f"go movetime {int(time_limit * 1000)}")
        
        start = time.time()
        while True:
            line = self._receive()
            if line.startswith("bestmove"):
                move_str = line.split()[1]
                return chess.Move.from_uci(move_str), time.time() - start
    
    def quit(self):
        """Quit engine"""
        self._send("quit")
        self.process.wait()

def play_test_game():
    """Play one game with 30s per move limit"""
    white = UCIEngine()
    black = UCIEngine()
    
    print(f"Playing {white.name} vs {black.name} with 30s per move limit\n")
    
    board = chess.Board()
    move_num = 1
    
    while not board.is_game_over() and move_num <= 50:
        engine = white if board.turn == chess.WHITE else black
        color = "White" if board.turn == chess.WHITE else "Black"
        
        print(f"Move {move_num} ({color})...", end='', flush=True)
        
        move, elapsed = engine.search(board, time_limit=30.0)
        
        print(f" {move.uci()} ({elapsed:.2f}s)", end='')
        
        if elapsed > 32.0:  # 2s grace period
            print(f" TIMEOUT! ({elapsed:.2f}s > 30s limit)")
            print(f"\nFAILURE: Engine exceeded time limit on move {move_num}")
            print(f"Position: {board.fen()}")
            white.quit()
            black.quit()
            return False
        else:
            print(f" OK")
        
        board.push(move)
        
        if board.turn == chess.BLACK:
            move_num += 1
    
    print(f"\nSUCCESS: Completed {move_num-1} moves without timeout!")
    print(f"Result: {board.result()}")
    white.quit()
    black.quit()
    return True

if __name__ == "__main__":
    import sys
    success = play_test_game()
    sys.exit(0 if success else 1)
