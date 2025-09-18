#!/usr/bin/env python3
"""
V7P3R Chess Engine - Self-Play Game Simulation
Tests adaptive evaluation and time management through complete games
"""

import sys
import os
import time
import chess
import chess.pgn
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

class GameTimer:
    """Manages time control for both players"""
    
    def __init__(self, time_control: str):
        self.time_control = time_control
        self.white_time_ms, self.black_time_ms, self.increment_ms = self._parse_time_control(time_control)
        self.white_remaining = self.white_time_ms
        self.black_remaining = self.black_time_ms
        
    def _parse_time_control(self, time_control: str) -> Tuple[int, int, int]:
        """Parse time control string into base time and increment"""
        if ':' in time_control:
            # Format like "10:5" (10 minutes + 5 second increment)
            parts = time_control.split(':')
            base_minutes = int(parts[0])
            increment_seconds = int(parts[1])
            base_ms = base_minutes * 60 * 1000
            increment_ms = increment_seconds * 1000
        else:
            # Format like "30" (30 minutes, no increment)
            base_minutes = int(time_control)
            base_ms = base_minutes * 60 * 1000
            increment_ms = 0
        
        return base_ms, base_ms, increment_ms
    
    def get_time_for_move(self, is_white: bool) -> int:
        """Get remaining time in milliseconds for the player to move"""
        return self.white_remaining if is_white else self.black_remaining
    
    def record_move_time(self, is_white: bool, time_used_ms: int):
        """Record time used for a move and apply increment"""
        if is_white:
            self.white_remaining = max(0, self.white_remaining - time_used_ms + self.increment_ms)
        else:
            self.black_remaining = max(0, self.black_remaining - time_used_ms + self.increment_ms)
    
    def is_time_up(self, is_white: bool) -> bool:
        """Check if player has run out of time"""
        remaining = self.white_remaining if is_white else self.black_remaining
        return remaining <= 0

class V7P3RSelfPlayGame:
    """Manages a self-play game between two instances of V7P3R"""
    
    def __init__(self, time_control: str = "10", max_moves: int = 200):
        self.time_control = time_control
        self.max_moves = max_moves
        self.timer = GameTimer(time_control)
        
        # Create two engine instances (could be same engine playing both sides)
        self.white_engine = V7P3REngine()
        self.black_engine = V7P3REngine()
        
        # Game state
        self.board = chess.Board()
        self.moves = []
        self.move_times = []
        self.evaluations = []
        self.posture_history = []
        
        # Performance tracking
        self.game_stats = {
            'total_nodes': 0,
            'total_time': 0.0,
            'move_count': 0,
            'time_control': time_control,
            'white_time_used': 0,
            'black_time_used': 0,
            'adaptive_eval_calls': 0,
            'posture_breakdown': {'emergency': 0, 'defensive': 0, 'balanced': 0, 'offensive': 0}
        }
    
    def play_game(self, verbose: bool = True) -> Dict:
        """Play a complete self-play game"""
        
        if verbose:
            print(f"Starting V7P3R Self-Play Game (Time Control: {self.time_control})")
            print("=" * 60)
        
        game_start_time = time.time()
        move_number = 1
        
        while not self.board.is_game_over() and len(self.moves) < self.max_moves:
            is_white = self.board.turn
            current_engine = self.white_engine if is_white else self.black_engine
            color_name = "White" if is_white else "Black"
            
            # Get remaining time
            remaining_time_ms = self.timer.get_time_for_move(is_white)
            
            # Check for time forfeit
            if self.timer.is_time_up(is_white):
                if verbose:
                    print(f"\n{color_name} loses on time!")
                self.game_stats['result'] = 'time_forfeit'
                self.game_stats['winner'] = 'Black' if is_white else 'White'
                break
            
            if verbose and move_number <= 10:  # Show first 10 moves in detail
                print(f"\nMove {move_number} - {color_name} to move")
                print(f"Position: {self.board.fen()}")
                print(f"Time remaining: {remaining_time_ms/1000:.1f}s")
            
            # Get position assessment
            volatility, posture = current_engine.posture_assessor.assess_position_posture(self.board)
            self.posture_history.append((move_number, color_name, volatility.value, posture.value))
            self.game_stats['posture_breakdown'][posture.value] += 1
            
            # Make move with time tracking
            move_start_time = time.time()
            
            try:
                # Set engine time parameters
                current_engine.current_time_remaining_ms = remaining_time_ms
                current_engine.current_moves_played = len(self.moves)
                
                # Determine search depth based on time remaining
                if remaining_time_ms > 300000:  # More than 5 minutes
                    search_depth = 5
                elif remaining_time_ms > 60000:  # More than 1 minute
                    search_depth = 4
                elif remaining_time_ms > 10000:  # More than 10 seconds
                    search_depth = 3
                else:  # Time pressure
                    search_depth = 2
                
                # Search for best move
                best_move = current_engine.search(self.board, depth=search_depth)
                
                if best_move is None or best_move not in self.board.legal_moves:
                    if verbose:
                        print(f"Engine returned invalid move: {best_move}")
                    break
                
                move_end_time = time.time()
                move_time_ms = int((move_end_time - move_start_time) * 1000)
                
                # Record move and time
                self.moves.append(best_move)
                self.move_times.append(move_time_ms)
                
                # Get evaluation of position after move
                self.board.push(best_move)
                position_eval = current_engine._evaluate_position(self.board)
                self.evaluations.append(position_eval)
                
                # Update time tracking
                self.timer.record_move_time(is_white, move_time_ms)
                if is_white:
                    self.game_stats['white_time_used'] += move_time_ms
                else:
                    self.game_stats['black_time_used'] += move_time_ms
                
                # Update game stats
                self.game_stats['total_nodes'] += current_engine.nodes_searched
                self.game_stats['total_time'] += (move_end_time - move_start_time)
                self.game_stats['move_count'] += 1
                
                # Get adaptive evaluation stats
                adaptive_stats = current_engine.adaptive_evaluator.get_evaluation_stats()
                self.game_stats['adaptive_eval_calls'] += adaptive_stats['calls']
                
                if verbose and move_number <= 10:
                    print(f"Move: {best_move}")
                    print(f"Time used: {move_time_ms}ms")
                    print(f"Nodes: {current_engine.nodes_searched:,}")
                    print(f"Position eval: {position_eval:.3f}")
                    print(f"Posture: {volatility.value}/{posture.value}")
                    print(f"Remaining time: {self.timer.get_time_for_move(not is_white)/1000:.1f}s")
                
                # Reset engine stats for next move
                current_engine.nodes_searched = 0
                current_engine.evaluation_cache.clear()
                
                # Increment move number for display
                if not is_white:  # After black's move
                    move_number += 1
                    
            except Exception as e:
                if verbose:
                    print(f"Error during move {move_number}: {e}")
                break
        
        # Game completed
        game_end_time = time.time()
        total_game_time = game_end_time - game_start_time
        
        # Determine result
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn else "White"  # Side that got checkmated loses
            result = "checkmate"
        elif self.board.is_stalemate():
            winner = "Draw"
            result = "stalemate"
        elif self.board.is_insufficient_material():
            winner = "Draw"
            result = "insufficient_material"
        elif len(self.moves) >= self.max_moves:
            winner = "Draw"
            result = "max_moves"
        else:
            winner = self.game_stats.get('winner', 'Unknown')
            result = self.game_stats.get('result', 'incomplete')
        
        # Finalize stats
        self.game_stats.update({
            'result': result,
            'winner': winner,
            'total_game_time': total_game_time,
            'moves_played': len(self.moves),
            'final_position': self.board.fen(),
            'white_time_remaining': self.timer.white_remaining,
            'black_time_remaining': self.timer.black_remaining,
            'avg_move_time': (self.game_stats['total_time'] / self.game_stats['move_count']) if self.game_stats['move_count'] > 0 else 0,
            'avg_nodes_per_move': (self.game_stats['total_nodes'] / self.game_stats['move_count']) if self.game_stats['move_count'] > 0 else 0
        })
        
        if verbose:
            self.print_game_summary()
        
        return self.game_stats
    
    def print_game_summary(self):
        """Print comprehensive game summary"""
        
        print(f"\n" + "=" * 60)
        print("GAME SUMMARY")
        print("=" * 60)
        
        # Basic game info
        print(f"Result: {self.game_stats['winner']} ({self.game_stats['result']})")
        print(f"Moves played: {self.game_stats['moves_played']}")
        print(f"Total game time: {self.game_stats['total_game_time']:.1f}s")
        print(f"Time control: {self.time_control}")
        
        # Time analysis
        print(f"\nTIME ANALYSIS:")
        print(f"White time used: {self.game_stats['white_time_used']/1000:.1f}s")
        print(f"Black time used: {self.game_stats['black_time_used']/1000:.1f}s")
        print(f"White time remaining: {self.game_stats['white_time_remaining']/1000:.1f}s")
        print(f"Black time remaining: {self.game_stats['black_time_remaining']/1000:.1f}s")
        print(f"Average move time: {self.game_stats['avg_move_time']*1000:.0f}ms")
        
        # Performance analysis
        print(f"\nPERFORMANCE ANALYSIS:")
        print(f"Total nodes: {self.game_stats['total_nodes']:,}")
        print(f"Average nodes per move: {self.game_stats['avg_nodes_per_move']:,.0f}")
        print(f"Adaptive eval calls: {self.game_stats['adaptive_eval_calls']:,}")
        
        # Posture analysis
        print(f"\nPOSTURE ANALYSIS:")
        total_postures = sum(self.game_stats['posture_breakdown'].values())
        for posture, count in self.game_stats['posture_breakdown'].items():
            percentage = (count / total_postures * 100) if total_postures > 0 else 0
            print(f"  {posture.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Recent move analysis
        if len(self.moves) >= 10:
            print(f"\nLAST 10 MOVES:")
            for i in range(max(0, len(self.moves)-10), len(self.moves)):
                move = self.moves[i]
                move_time = self.move_times[i]
                evaluation = self.evaluations[i]
                move_num = (i // 2) + 1
                color = "White" if i % 2 == 0 else "Black"
                print(f"  {move_num}. {move} ({color}) - {move_time}ms, eval: {evaluation:.3f}")
        
        # Print final position
        print(f"\nFINAL POSITION:")
        print(f"FEN: {self.game_stats['final_position']}")
    
    def export_pgn(self, filename: Optional[str] = None) -> str:
        """Export game as PGN"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"v7p3r_selfplay_{self.time_control}_{timestamp}.pgn"
        
        # Create PGN game
        game = chess.pgn.Game()
        game.headers["Event"] = "V7P3R Self-Play Test"
        game.headers["Site"] = "Local"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = "V7P3R v11 Phase 3B"
        game.headers["Black"] = "V7P3R v11 Phase 3B"
        game.headers["TimeControl"] = self.time_control
        game.headers["Result"] = "1-0" if self.game_stats['winner'] == "White" else "0-1" if self.game_stats['winner'] == "Black" else "1/2-1/2"
        
        # Add moves
        node = game
        temp_board = chess.Board()
        
        for move in self.moves:
            node = node.add_variation(move)
            temp_board.push(move)
        
        # Save PGN
        with open(filename, 'w') as f:
            f.write(str(game))
        
        return filename

def run_time_control_tests():
    """Run self-play tests with different time controls"""
    
    time_controls = ["30", "10", "10:5", "5:5", "2:1", "1:1"]
    results = {}
    
    print("V7P3R Time Control Scaling Tests")
    print("=" * 60)
    
    for time_control in time_controls:
        print(f"\nTesting time control: {time_control}")
        print("-" * 40)
        
        try:
            game = V7P3RSelfPlayGame(time_control, max_moves=100)  # Shorter games for testing
            stats = game.play_game(verbose=False)
            results[time_control] = stats
            
            print(f"✓ {time_control}: {stats['winner']} in {stats['moves_played']} moves")
            print(f"  Avg move time: {stats['avg_move_time']*1000:.0f}ms")
            print(f"  Avg nodes/move: {stats['avg_nodes_per_move']:,.0f}")
            
            # Export PGN
            pgn_file = game.export_pgn()
            print(f"  PGN saved: {pgn_file}")
            
        except Exception as e:
            print(f"✗ {time_control}: Failed - {e}")
            results[time_control] = {'error': str(e)}
    
    # Summary analysis
    print(f"\n" + "=" * 60)
    print("TIME CONTROL ANALYSIS SUMMARY")
    print("=" * 60)
    
    for tc, stats in results.items():
        if 'error' not in stats:
            print(f"{tc:>6}: {stats['avg_move_time']*1000:6.0f}ms avg, {stats['avg_nodes_per_move']:8,.0f} nodes/move")
    
    return results

if __name__ == "__main__":
    print("V7P3R Self-Play Game Simulation")
    print("=" * 60)
    
    # Single detailed game
    print("\n1. DETAILED SELF-PLAY GAME (2:1 time control)")
    game = V7P3RSelfPlayGame("2:1", max_moves=150)
    stats = game.play_game(verbose=True)
    pgn_file = game.export_pgn()
    print(f"\nGame PGN saved as: {pgn_file}")
    
    # Time control scaling tests
    print(f"\n" + "=" * 60)
    print("2. TIME CONTROL SCALING TESTS")
    scaling_results = run_time_control_tests()
    
    print(f"\n✓ Self-play testing completed!")
    print("Check generated PGN files for detailed game analysis.")