#!/usr/bin/env python3
"""
V16.2 Lichess Deployment Monitoring Script
Step 1: Validate v16.2's depth fix on Lichess

Monitors first 20-50 games for:
1. Actual search depth reached (target: consistent 5-8, not 2-10 fluctuating)
2. Material blunders
3. Positional coherence
4. ELO progression vs v14.1 baseline (1496)

This validates if the null move pruning bug fix is working in live play.
"""

import os
import re
import chess
import chess.pgn
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import json

class V162LichessMonitor:
    """Monitor v16.2 deployment on Lichess"""
    
    def __init__(self):
        self.records_dir = Path("s:/Programming/Chess Engines/Deployed Engines/v7p3r-lichess-engine/game_records")
        self.v14_1_baseline_elo = 1496
        
    def analyze_game_records(self, max_games: int = 50) -> Dict:
        """Analyze recent game records from Lichess"""
        
        print("=" * 80)
        print("V16.2 LICHESS DEPLOYMENT MONITOR")
        print("=" * 80)
        print(f"Target: Validate depth bug fix in first {max_games} games")
        print(f"Baseline: v14.1 @ {self.v14_1_baseline_elo} ELO")
        print()
        
        if not self.records_dir.exists():
            print(f"❌ Game records directory not found: {self.records_dir}")
            return {}
        
        # Collect all PGN files
        pgn_files = sorted(self.records_dir.glob("*.pgn"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        stats = {
            'total_games': 0,
            'completed_games': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'depth_stats': [],
            'material_blunders': [],
            'tactical_blunders': [],
            'ratings': [],
            'game_details': [],
            'time_per_move': [],
            'nps_stats': []
        }
        
        games_analyzed = 0
        
        for pgn_file in pgn_files:
            if games_analyzed >= max_games:
                break
                
            try:
                with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
                    while games_analyzed < max_games:
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break
                        
                        # Check if this is a v7p3r_bot game
                        white = game.headers.get("White", "")
                        black = game.headers.get("Black", "")
                        
                        if "v7p3r" not in white.lower() and "v7p3r" not in black.lower():
                            continue
                        
                        v7p3r_color = "white" if "v7p3r" in white.lower() else "black"
                        opponent = black if v7p3r_color == "white" else white
                        
                        # Extract game info
                        result = game.headers.get("Result", "*")
                        date = game.headers.get("Date", "Unknown")
                        time_control = game.headers.get("TimeControl", "Unknown")
                        
                        stats['total_games'] += 1
                        
                        if result == "*":
                            continue  # Skip ongoing games
                        
                        stats['completed_games'] += 1
                        games_analyzed += 1
                        
                        # Count results
                        if (result == "1-0" and v7p3r_color == "white") or (result == "0-1" and v7p3r_color == "black"):
                            stats['wins'] += 1
                            game_result = "WIN"
                        elif (result == "0-1" and v7p3r_color == "white") or (result == "1-0" and v7p3r_color == "black"):
                            stats['losses'] += 1
                            game_result = "LOSS"
                        else:
                            stats['draws'] += 1
                            game_result = "DRAW"
                        
                        # Extract rating
                        if v7p3r_color == "white":
                            v7p3r_rating = game.headers.get("WhiteElo", "?")
                        else:
                            v7p3r_rating = game.headers.get("BlackElo", "?")
                        
                        if v7p3r_rating != "?":
                            stats['ratings'].append(int(v7p3r_rating))
                        
                        # Analyze moves for depth and quality
                        depth_info = self._analyze_game_moves(game, v7p3r_color)
                        
                        game_detail = {
                            'opponent': opponent,
                            'result': game_result,
                            'rating': v7p3r_rating,
                            'date': date,
                            'time_control': time_control,
                            'depth_avg': depth_info['avg_depth'],
                            'depth_min': depth_info['min_depth'],
                            'depth_max': depth_info['max_depth'],
                            'depth_consistency': depth_info['consistency'],
                            'material_blunders': depth_info['material_blunders'],
                            'tactical_issues': depth_info['tactical_issues']
                        }
                        
                        stats['game_details'].append(game_detail)
                        
                        if depth_info['depths']:
                            stats['depth_stats'].extend(depth_info['depths'])
                        
                        print(f"[{games_analyzed}/{max_games}] {game_result} vs {opponent} | "
                              f"ELO: {v7p3r_rating} | "
                              f"Depth: {depth_info['avg_depth']:.1f} (min: {depth_info['min_depth']}, max: {depth_info['max_depth']}) | "
                              f"Consistency: {depth_info['consistency']}")
                        
            except Exception as e:
                print(f"Error reading {pgn_file}: {e}")
                continue
        
        return stats
    
    def _analyze_game_moves(self, game: chess.pgn.Game, v7p3r_color: str) -> Dict:
        """Analyze individual game moves for depth and quality"""
        
        depths = []
        material_blunders = 0
        tactical_issues = 0
        
        board = game.board()
        node = game
        move_num = 0
        
        while node.variations:
            node = node.variation(0)
            move_num += 1
            
            # Only analyze v7p3r's moves
            if (v7p3r_color == "white" and board.turn == chess.WHITE) or \
               (v7p3r_color == "black" and board.turn == chess.BLACK):
                
                # Extract depth from comment if available
                comment = node.comment
                if comment:
                    depth_match = re.search(r'depth[:\s]+(\d+)', comment, re.IGNORECASE)
                    if depth_match:
                        depth = int(depth_match.group(1))
                        depths.append(depth)
                    
                    # Check for evaluation drops (potential blunders)
                    eval_match = re.search(r'eval[:\s]+([-+]?\d+\.?\d*)', comment, re.IGNORECASE)
                    if eval_match:
                        eval_score = float(eval_match.group(1))
                        if abs(eval_score) > 3.0:  # Significant material loss
                            material_blunders += 1
                
                # Basic material analysis
                move = node.move
                if move:
                    # Check if move hangs material (simplified check)
                    if board.is_capture(move):
                        # Check if this is a bad capture
                        piece_taken = board.piece_at(move.to_square)
                        piece_moving = board.piece_at(move.from_square)
                        
                        if piece_taken and piece_moving:
                            # Simple material check
                            piece_values = {
                                chess.PAWN: 1,
                                chess.KNIGHT: 3,
                                chess.BISHOP: 3,
                                chess.ROOK: 5,
                                chess.QUEEN: 9,
                                chess.KING: 0
                            }
                            
                            taken_value = piece_values.get(piece_taken.piece_type, 0)
                            moving_value = piece_values.get(piece_moving.piece_type, 0)
                            
                            board.push(move)
                            # Check if piece is hanging after capture
                            if board.is_attacked_by(not board.turn, move.to_square):
                                if moving_value > taken_value + 1:
                                    tactical_issues += 1
                            board.pop()
            
            if move_num < len(node.parent.variations):
                board.push(node.move)
        
        # Calculate stats
        avg_depth = sum(depths) / len(depths) if depths else 0
        min_depth = min(depths) if depths else 0
        max_depth = max(depths) if depths else 0
        
        # Consistency score: lower std deviation = more consistent
        if len(depths) > 1:
            variance = sum((d - avg_depth) ** 2 for d in depths) / len(depths)
            std_dev = variance ** 0.5
            consistency = f"{std_dev:.1f} std dev"
        else:
            consistency = "insufficient data"
        
        return {
            'depths': depths,
            'avg_depth': avg_depth,
            'min_depth': min_depth,
            'max_depth': max_depth,
            'consistency': consistency,
            'material_blunders': material_blunders,
            'tactical_issues': tactical_issues
        }
    
    def generate_report(self, stats: Dict):
        """Generate comprehensive monitoring report"""
        
        print("\n" + "=" * 80)
        print("V16.2 DEPLOYMENT VALIDATION REPORT")
        print("=" * 80)
        
        if stats['completed_games'] == 0:
            print("❌ No completed games found. Deploy v16.2 and play games first.")
            return
        
        # Performance Summary
        print("\n📊 PERFORMANCE SUMMARY")
        print("-" * 80)
        print(f"Games Analyzed: {stats['completed_games']}")
        print(f"Wins: {stats['wins']} ({stats['wins']/stats['completed_games']*100:.1f}%)")
        print(f"Losses: {stats['losses']} ({stats['losses']/stats['completed_games']*100:.1f}%)")
        print(f"Draws: {stats['draws']} ({stats['draws']/stats['completed_games']*100:.1f}%)")
        
        score = stats['wins'] + (stats['draws'] * 0.5)
        score_pct = score / stats['completed_games'] * 100
        print(f"Score Rate: {score_pct:.1f}%")
        
        if score_pct < 45:
            print("🔴 CRITICAL: Performance below 45% - major regression!")
        elif score_pct < 50:
            print("🟠 WARNING: Performance below 50% - investigate immediately")
        elif score_pct < 60:
            print("🟡 ACCEPTABLE: Performance 50-60% - monitoring needed")
        else:
            print("🟢 STRONG: Performance above 60% - looking good!")
        
        # Rating Analysis
        if stats['ratings']:
            print("\n🏆 RATING PROGRESSION")
            print("-" * 80)
            current_elo = stats['ratings'][-1] if stats['ratings'] else "Unknown"
            avg_elo = sum(stats['ratings']) / len(stats['ratings'])
            
            print(f"Current ELO: {current_elo}")
            print(f"Average ELO: {avg_elo:.0f}")
            print(f"v14.1 Baseline: {self.v14_1_baseline_elo}")
            
            if isinstance(current_elo, int):
                elo_delta = current_elo - self.v14_1_baseline_elo
                if elo_delta > 0:
                    print(f"🟢 +{elo_delta} improvement over v14.1")
                elif elo_delta < -50:
                    print(f"🔴 {elo_delta} regression - ROLLBACK CANDIDATE")
                else:
                    print(f"🟡 {elo_delta} slight regression - monitor closely")
        
        # Depth Analysis (CRITICAL)
        if stats['depth_stats']:
            print("\n🎯 DEPTH ANALYSIS (CRITICAL)")
            print("-" * 80)
            avg_depth = sum(stats['depth_stats']) / len(stats['depth_stats'])
            min_depth = min(stats['depth_stats'])
            max_depth = max(stats['depth_stats'])
            
            # Count depth distribution
            depth_2_3 = sum(1 for d in stats['depth_stats'] if d <= 3)
            depth_4_5 = sum(1 for d in stats['depth_stats'] if 4 <= d <= 5)
            depth_6_8 = sum(1 for d in stats['depth_stats'] if 6 <= d <= 8)
            depth_9_plus = sum(1 for d in stats['depth_stats'] if d >= 9)
            
            print(f"Average Depth: {avg_depth:.1f}")
            print(f"Depth Range: {min_depth} - {max_depth}")
            print(f"\nDepth Distribution:")
            print(f"  Depth 2-3 (BUG RANGE): {depth_2_3} moves ({depth_2_3/len(stats['depth_stats'])*100:.1f}%)")
            print(f"  Depth 4-5: {depth_4_5} moves ({depth_4_5/len(stats['depth_stats'])*100:.1f}%)")
            print(f"  Depth 6-8 (TARGET): {depth_6_8} moves ({depth_6_8/len(stats['depth_stats'])*100:.1f}%)")
            print(f"  Depth 9+ (EXCELLENT): {depth_9_plus} moves ({depth_9_plus/len(stats['depth_stats'])*100:.1f}%)")
            
            # Critical assessment
            if depth_2_3 > len(stats['depth_stats']) * 0.3:
                print("\n🔴 CRITICAL: >30% moves at depth 2-3 - BUG NOT FIXED!")
            elif depth_2_3 > len(stats['depth_stats']) * 0.1:
                print("\n🟠 WARNING: >10% moves at depth 2-3 - partial bug fix?")
            elif depth_6_8 > len(stats['depth_stats']) * 0.5:
                print("\n🟢 EXCELLENT: >50% moves at target depth 6-8 - bug fixed!")
            else:
                print("\n🟡 MODERATE: Bug appears fixed but inconsistent depth")
            
            # Consistency check
            variance = sum((d - avg_depth) ** 2 for d in stats['depth_stats']) / len(stats['depth_stats'])
            std_dev = variance ** 0.5
            print(f"\nDepth Consistency: {std_dev:.1f} std deviation")
            
            if std_dev < 1.5:
                print("🟢 EXCELLENT: Very consistent depth (like PositionalOpponent)")
            elif std_dev < 2.5:
                print("🟡 GOOD: Reasonably consistent depth")
            else:
                print("🔴 POOR: Inconsistent depth - reliability issues")
        else:
            print("\n🔴 WARNING: No depth data found in game comments")
            print("   Ensure v16.2 is writing depth info to PGN comments")
        
        # Material Blunders
        total_material_blunders = sum(g['material_blunders'] for g in stats['game_details'])
        total_tactical_issues = sum(g['tactical_issues'] for g in stats['game_details'])
        
        print(f"\n⚠️  QUALITY METRICS")
        print("-" * 80)
        print(f"Material Blunders: {total_material_blunders}")
        print(f"Tactical Issues: {total_tactical_issues}")
        
        if total_material_blunders == 0:
            print("🟢 EXCELLENT: No material blunders detected")
        elif total_material_blunders < stats['completed_games'] * 0.1:
            print("🟡 GOOD: Minimal material blunders (<10% games)")
        else:
            print("🔴 CONCERN: Frequent material blunders - check move filtering")
        
        # Recent Games Detail
        print(f"\n📋 RECENT GAMES DETAIL (Last 10)")
        print("-" * 80)
        for game in stats['game_details'][-10:]:
            print(f"{game['result']:4} vs {game['opponent']:20} | "
                  f"ELO: {game['rating']:4} | "
                  f"Depth: {game['depth_avg']:4.1f} ({game['depth_min']}-{game['depth_max']}) | "
                  f"Blunders: {game['material_blunders']}")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS")
        print("-" * 80)
        
        if stats['completed_games'] < 20:
            print("⏳ CONTINUE MONITORING: Need 20+ games for reliable assessment")
        elif score_pct < 45 or (stats['depth_stats'] and avg_depth < 4):
            print("🔴 ROLLBACK to v14.1: Performance regression too severe")
        elif score_pct > 55 and (not stats['depth_stats'] or avg_depth >= 5):
            print("🟢 CONTINUE v16.2: Performance acceptable, continue monitoring")
        else:
            print("🟡 CAUTIOUS MONITORING: Borderline performance, collect more data")
        
        # Save report to file
        self._save_report_json(stats)
    
    def _save_report_json(self, stats: Dict):
        """Save report to JSON for historical tracking"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path("s:/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/docs") / f"v16_2_lichess_monitoring_{timestamp}.json"
        
        # Prepare JSON-serializable stats
        json_stats = {
            'timestamp': timestamp,
            'completed_games': stats['completed_games'],
            'wins': stats['wins'],
            'losses': stats['losses'],
            'draws': stats['draws'],
            'score_rate': (stats['wins'] + stats['draws'] * 0.5) / stats['completed_games'] * 100 if stats['completed_games'] > 0 else 0,
            'ratings': stats['ratings'],
            'avg_rating': sum(stats['ratings']) / len(stats['ratings']) if stats['ratings'] else 0,
            'depth_stats': {
                'all_depths': stats['depth_stats'],
                'avg_depth': sum(stats['depth_stats']) / len(stats['depth_stats']) if stats['depth_stats'] else 0,
                'min_depth': min(stats['depth_stats']) if stats['depth_stats'] else 0,
                'max_depth': max(stats['depth_stats']) if stats['depth_stats'] else 0
            },
            'game_details': stats['game_details']
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(json_stats, f, indent=2)
            print(f"\n💾 Report saved: {report_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save report: {e}")

def main():
    """Main monitoring function"""
    
    monitor = V162LichessMonitor()
    
    print("Starting v16.2 Lichess deployment monitoring...")
    print("This will analyze recent games to validate the depth bug fix.\n")
    
    # Analyze first 50 games
    stats = monitor.analyze_game_records(max_games=50)
    
    if stats:
        monitor.generate_report(stats)
    else:
        print("❌ No game data available for analysis")
        print("\nNext steps:")
        print("1. Ensure v16.2 is deployed to Lichess")
        print("2. Play 20-50 games")
        print("3. Run this script again")

if __name__ == "__main__":
    main()
