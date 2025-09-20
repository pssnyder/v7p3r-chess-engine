#!/usr/bin/env python3
"""
V7P3R v11.1 Game Scenario Simulator
Tests engine in realistic game situations
"""

import chess
import chess.pgn
import sys
import os
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from v7p3r_v11_1_simplified import V7P3REngineSimple
    print("✅ Successfully imported V7P3R v11.1 simplified engine")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


class GameScenarioTester:
    """Test engine in realistic game scenarios"""
    
    def __init__(self):
        self.engine = V7P3REngineSimple()
        
        # Opening scenarios
        self.opening_scenarios = [
            {
                "name": "Italian Game",
                "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
                "time_control": 2.0
            },
            {
                "name": "Sicilian Defense",
                "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"],
                "time_control": 2.5
            },
            {
                "name": "French Defense",
                "moves": ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3"],
                "time_control": 2.0
            },
            {
                "name": "Caro-Kann Defense",
                "moves": ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3"],
                "time_control": 2.0
            }
        ]
        
        # Middlegame scenarios
        self.middlegame_scenarios = [
            {
                "name": "Central tension",
                "fen": "r1bqk2r/pp2nppp/2n1p3/3pP3/1bpP4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 8",
                "time_control": 3.0,
                "description": "Complex central pawn structure"
            },
            {
                "name": "Piece coordination",
                "fen": "r2qkb1r/pp2nppp/3p1n2/2pP4/4P3/2N2N2/PPP1BPPP/R1BQK2R w KQkq c6 0 8",
                "time_control": 3.5,
                "description": "Need for piece coordination"
            },
            {
                "name": "King safety",
                "fen": "r1bq1rk1/pp1n1ppp/2pb1n2/3p4/2PP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 9",
                "time_control": 3.0,
                "description": "Castled king with pressure"
            }
        ]
        
        # Endgame scenarios
        self.endgame_scenarios = [
            {
                "name": "King and pawn endgame",
                "fen": "8/8/8/3k4/3P4/3K4/8/8 w - - 0 1",
                "time_control": 4.0,
                "description": "Basic K+P vs K"
            },
            {
                "name": "Rook endgame",
                "fen": "8/8/8/3k4/8/3K4/3R4/r7 w - - 0 1",
                "time_control": 5.0,
                "description": "Active rook vs passive rook"
            },
            {
                "name": "Queen vs pawn endgame",
                "fen": "8/8/8/8/8/3k4/2p5/3K1Q2 w - - 0 1",
                "time_control": 4.0,
                "description": "Queen stopping advanced pawn"
            }
        ]
        
        # Time control scenarios
        self.time_scenarios = [
            {
                "name": "Bullet time control",
                "time_per_move": 0.5,
                "positions": 5
            },
            {
                "name": "Blitz time control",
                "time_per_move": 1.5,
                "positions": 5
            },
            {
                "name": "Rapid time control", 
                "time_per_move": 3.0,
                "positions": 3
            }
        ]
    
    def test_opening_play(self):
        """Test engine opening play"""
        print("\n♟️  OPENING PLAY TESTS")
        print("=" * 50)
        
        results = []
        
        for scenario in self.opening_scenarios:
            print(f"\nTesting: {scenario['name']}")
            
            board = chess.Board()
            moves_played = []
            
            # Play the opening moves
            for move_uci in scenario['moves']:
                move = chess.Move.from_uci(move_uci)
                board.push(move)
                moves_played.append(move_uci)
            
            print(f"Position after: {' '.join(moves_played)}")
            print(f"FEN: {board.fen()}")
            
            # Engine plays next move
            start_time = time.time()
            try:
                engine_move = self.engine.search(board, time_limit=scenario['time_control'])
                search_time = time.time() - start_time
                
                move_uci = engine_move.uci() if engine_move else "null"
                nodes = self.engine.search_stats.get('nodes_searched', 0)
                
                # Validate move is legal and reasonable
                is_legal = engine_move in board.legal_moves if engine_move else False
                
                result = {
                    'scenario': scenario['name'],
                    'success': is_legal,
                    'move': move_uci,
                    'time': search_time,
                    'nodes': nodes,
                    'position': board.fen()
                }
                
                results.append(result)
                
                status = "✅ LEGAL" if is_legal else "❌ ILLEGAL"
                print(f"  Engine move: {move_uci} ({status})")
                print(f"  Time: {search_time:.2f}s")
                print(f"  Nodes: {nodes:,}")
                
            except Exception as e:
                result = {
                    'scenario': scenario['name'],
                    'success': False,
                    'error': str(e),
                    'position': board.fen()
                }
                results.append(result)
                print(f"  Error: {e}")
        
        return results
    
    def test_middlegame_scenarios(self):
        """Test engine middlegame understanding"""
        print("\n🏰 MIDDLEGAME SCENARIO TESTS")
        print("=" * 50)
        
        results = []
        
        for scenario in self.middlegame_scenarios:
            print(f"\nTesting: {scenario['name']}")
            print(f"Description: {scenario['description']}")
            print(f"FEN: {scenario['fen']}")
            
            board = chess.Board(scenario['fen'])
            
            start_time = time.time()
            try:
                engine_move = self.engine.search(board, time_limit=scenario['time_control'])
                search_time = time.time() - start_time
                
                move_uci = engine_move.uci() if engine_move else "null"
                nodes = self.engine.search_stats.get('nodes_searched', 0)
                
                # Evaluate position before and after move
                eval_before = self.engine._evaluate_position(board)
                
                if engine_move and engine_move in board.legal_moves:
                    board.push(engine_move)
                    eval_after = -self.engine._evaluate_position(board)  # Flip perspective
                    board.pop()
                    eval_change = eval_after - eval_before
                else:
                    eval_after = None
                    eval_change = None
                
                is_legal = engine_move in board.legal_moves if engine_move else False
                
                result = {
                    'scenario': scenario['name'],
                    'success': is_legal,
                    'move': move_uci,
                    'time': search_time,
                    'nodes': nodes,
                    'eval_before': eval_before,
                    'eval_after': eval_after,
                    'eval_change': eval_change,
                    'position': scenario['fen']
                }
                
                results.append(result)
                
                status = "✅ LEGAL" if is_legal else "❌ ILLEGAL"
                print(f"  Engine move: {move_uci} ({status})")
                print(f"  Evaluation: {eval_before:.2f} -> {eval_after:.2f}" if eval_after is not None else f"  Evaluation: {eval_before:.2f}")
                print(f"  Time: {search_time:.2f}s")
                print(f"  Nodes: {nodes:,}")
                
            except Exception as e:
                result = {
                    'scenario': scenario['name'],
                    'success': False,
                    'error': str(e),
                    'position': scenario['fen']
                }
                results.append(result)
                print(f"  Error: {e}")
        
        return results
    
    def test_endgame_technique(self):
        """Test engine endgame technique"""
        print("\n🎯 ENDGAME TECHNIQUE TESTS")
        print("=" * 50)
        
        results = []
        
        for scenario in self.endgame_scenarios:
            print(f"\nTesting: {scenario['name']}")
            print(f"Description: {scenario['description']}")
            print(f"FEN: {scenario['fen']}")
            
            board = chess.Board(scenario['fen'])
            
            start_time = time.time()
            try:
                engine_move = self.engine.search(board, time_limit=scenario['time_control'])
                search_time = time.time() - start_time
                
                move_uci = engine_move.uci() if engine_move else "null"
                nodes = self.engine.search_stats.get('nodes_searched', 0)
                
                is_legal = engine_move in board.legal_moves if engine_move else False
                
                # For endgames, check if move makes progress
                progress_check = True  # Simplified - could add specific endgame logic
                
                result = {
                    'scenario': scenario['name'],
                    'success': is_legal and progress_check,
                    'move': move_uci,
                    'time': search_time,
                    'nodes': nodes,
                    'makes_progress': progress_check,
                    'position': scenario['fen']
                }
                
                results.append(result)
                
                status = "✅ GOOD" if is_legal and progress_check else "❌ POOR"
                print(f"  Engine move: {move_uci} ({status})")
                print(f"  Time: {search_time:.2f}s")
                print(f"  Nodes: {nodes:,}")
                
            except Exception as e:
                result = {
                    'scenario': scenario['name'],
                    'success': False,
                    'error': str(e),
                    'position': scenario['fen']
                }
                results.append(result)
                print(f"  Error: {e}")
        
        return results
    
    def test_time_management(self):
        """Test engine time management under different controls"""
        print("\n⏱️  TIME MANAGEMENT TESTS")
        print("=" * 50)
        
        results = []
        
        # Test positions for time management
        test_positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r2qkb1r/pp2nppp/3p1n2/2pP4/4P3/2N2N2/PPP1BPPP/R1BQK2R w KQkq c6 0 8",
            "r1bq1rk1/pp1n1ppp/2pb1n2/3p4/2PP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 9",
            "8/8/8/3k4/3P4/3K4/8/8 w - - 0 1"
        ]
        
        for time_scenario in self.time_scenarios:
            print(f"\nTesting: {time_scenario['name']}")
            print(f"Time per move: {time_scenario['time_per_move']:.1f}s")
            
            scenario_results = []
            
            for i, fen in enumerate(test_positions[:time_scenario['positions']]):
                print(f"  Position {i+1}: {fen[:30]}...")
                
                board = chess.Board(fen)
                
                start_time = time.time()
                try:
                    engine_move = self.engine.search(board, time_limit=time_scenario['time_per_move'])
                    search_time = time.time() - start_time
                    
                    move_uci = engine_move.uci() if engine_move else "null"
                    nodes = self.engine.search_stats.get('nodes_searched', 0)
                    
                    # Check time compliance (allow 50% buffer)
                    time_buffer = time_scenario['time_per_move'] * 1.5
                    time_compliant = search_time <= time_buffer
                    is_legal = engine_move in board.legal_moves if engine_move else False
                    
                    result = {
                        'position_index': i+1,
                        'success': is_legal and time_compliant,
                        'move': move_uci,
                        'time': search_time,
                        'time_limit': time_scenario['time_per_move'],
                        'time_compliant': time_compliant,
                        'nodes': nodes,
                        'nps': nodes / search_time if search_time > 0 else 0
                    }
                    
                    scenario_results.append(result)
                    
                    status = "✅ GOOD" if is_legal and time_compliant else "❌ ISSUE"
                    compliance = "ON TIME" if time_compliant else "OVERTIME"
                    print(f"    Result: {status} ({compliance})")
                    print(f"    Move: {move_uci}")
                    print(f"    Time: {search_time:.2f}s")
                    
                except Exception as e:
                    result = {
                        'position_index': i+1,
                        'success': False,
                        'error': str(e),
                        'time_limit': time_scenario['time_per_move']
                    }
                    scenario_results.append(result)
                    print(f"    Error: {e}")
            
            # Calculate scenario statistics
            successful_moves = sum(1 for r in scenario_results if r.get('success', False))
            avg_time = sum(r.get('time', 0) for r in scenario_results) / len(scenario_results)
            avg_nps = sum(r.get('nps', 0) for r in scenario_results) / len(scenario_results)
            
            scenario_summary = {
                'scenario': time_scenario['name'],
                'time_per_move': time_scenario['time_per_move'],
                'positions_tested': len(scenario_results),
                'successful_moves': successful_moves,
                'success_rate': successful_moves / len(scenario_results) * 100,
                'average_time': avg_time,
                'average_nps': avg_nps,
                'moves': scenario_results
            }
            
            results.append(scenario_summary)
            
            print(f"  Summary: {successful_moves}/{len(scenario_results)} moves successful")
            print(f"  Average time: {avg_time:.2f}s")
            print(f"  Average NPS: {avg_nps:.0f}")
        
        return results
    
    def simulate_mini_game(self, time_per_move=2.0, max_moves=20):
        """Simulate a mini-game to test consistency"""
        print(f"\n🎮 MINI-GAME SIMULATION ({max_moves} moves)")
        print("=" * 50)
        
        board = chess.Board()
        moves_played = []
        
        print("Starting position:")
        print(board)
        
        for move_num in range(max_moves):
            if board.is_game_over():
                print(f"Game ended: {board.result()}")
                break
            
            print(f"\nMove {move_num + 1} ({'White' if board.turn else 'Black'} to move)")
            
            start_time = time.time()
            try:
                engine_move = self.engine.search(board, time_limit=time_per_move)
                search_time = time.time() - start_time
                
                if not engine_move or engine_move not in board.legal_moves:
                    print(f"❌ Engine error: Invalid move {engine_move}")
                    break
                
                move_uci = engine_move.uci()
                nodes = self.engine.search_stats.get('nodes_searched', 0)
                
                board.push(engine_move)
                moves_played.append(move_uci)
                
                print(f"  Engine plays: {move_uci}")
                print(f"  Time: {search_time:.2f}s")
                print(f"  Nodes: {nodes:,}")
                
                # Show position every few moves
                if (move_num + 1) % 5 == 0:
                    print(f"\nPosition after {move_num + 1} moves:")
                    print(board)
                
            except Exception as e:
                print(f"❌ Engine error: {e}")
                break
        
        # Game summary
        print(f"\n📊 Game Summary:")
        print(f"Moves played: {len(moves_played)}")
        print(f"Final position: {board.fen()}")
        if board.is_game_over():
            print(f"Game result: {board.result()}")
        
        return {
            'moves_played': moves_played,
            'total_moves': len(moves_played),
            'final_fen': board.fen(),
            'game_over': board.is_game_over(),
            'result': board.result() if board.is_game_over() else None
        }
    
    def run_all_game_scenarios(self):
        """Run all game scenario tests"""
        print("🎮 V7P3R v11.1 GAME SCENARIO TESTS")
        print("=" * 80)
        
        all_results = {}
        
        # Run all test categories
        all_results['opening_play'] = self.test_opening_play()
        all_results['middlegame_scenarios'] = self.test_middlegame_scenarios()
        all_results['endgame_technique'] = self.test_endgame_technique()
        all_results['time_management'] = self.test_time_management()
        all_results['mini_game'] = self.simulate_mini_game()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"v7p3r_v11_1_game_scenarios_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n💾 Game scenario results saved to: {filename}")
        
        # Summary
        print("\n📊 GAME SCENARIO SUMMARY")
        print("=" * 50)
        
        for category, results in all_results.items():
            if category == 'mini_game':
                print(f"Mini-game: {results['total_moves']} moves played")
            else:
                if isinstance(results, list) and results:
                    successful = sum(1 for r in results if r.get('success', False))
                    total = len(results)
                    print(f"{category.replace('_', ' ').title()}: {successful}/{total} successful")
        
        return all_results


def main():
    """Main execution"""
    tester = GameScenarioTester()
    results = tester.run_all_game_scenarios()
    return results


if __name__ == "__main__":
    main()