#!/usr/bin/env python3
"""
Create test data in the metrics database to verify dashboard functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metrics.metrics_store import MetricsStore
from datetime import datetime, timedelta
import random

def create_test_data():
    """Create test data in the metrics database"""
    print("Creating test data for metrics dashboard...")
    
    metrics_store = MetricsStore()
    
    # Create several test games with v7p3r data
    for i in range(5):
        timestamp = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d_%H%M%S")
        game_id = f"test_game_{timestamp}"
        
        # Randomly determine winner
        winners = ["1-0", "0-1", "1/2-1/2"]
        winner = random.choice(winners)
        
        # Create game result with v7p3r engines
        metrics_store.add_game_result(
            game_id=game_id,
            timestamp=timestamp,
            winner=winner,
            game_pgn=f"[Event 'Test Game {i}']\n[Result '{winner}']\n1. e4 e5 *",
            white_player="AI: v7p3r (Depth 4)",
            black_player="AI: Stockfish (Elo 1500)",
            game_length=random.randint(20, 60),
            white_engine_config={
                'engine': 'v7p3r',
                'engine_type': 'deepsearch',
                'depth': 4,
                'exclude_from_metrics': False
            },
            black_engine_config={
                'engine': 'stockfish',
                'engine_type': 'stockfish',
                'depth': 3,
                'exclude_from_metrics': True
            }
        )
        
        # Add move metrics for this game
        for move_num in range(1, random.randint(10, 20)):
            # White move (v7p3r)
            metrics_store.add_move_metric(
                game_id=game_id,
                move_number=move_num,
                player_color='w',
                move_uci=f'e{2+move_num}e{3+move_num}',
                fen_before='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                evaluation=random.uniform(-0.5, 0.5),
                engine_type='deepsearch',
                depth=4,
                nodes_searched=random.randint(1000, 5000),
                time_taken=random.uniform(0.1, 2.0),
                pv_line=f'e{2+move_num}e{3+move_num} d7d6'
            )
            
            # Black move (Stockfish - excluded)
            metrics_store.add_move_metric(
                game_id=game_id,
                move_number=move_num,
                player_color='b',
                move_uci=f'd{7-move_num}d{6-move_num}',
                fen_before='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1',
                evaluation=random.uniform(-0.5, 0.5),
                engine_type='stockfish',
                depth=3,
                nodes_searched=random.randint(2000, 8000),
                time_taken=random.uniform(0.1, 1.5),
                pv_line=f'd{7-move_num}d{6-move_num} Nf3'
            )
    
    print(f"Created {5} test games with move metrics")
    
    # Check what was created
    connection = metrics_store._get_connection()
    with connection:
        cursor = connection.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM game_results")
        game_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM move_metrics")
        move_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM move_metrics WHERE engine_type = 'deepsearch'")
        v7p3r_moves = cursor.fetchone()[0]
        
        print(f"Database now contains:")
        print(f"  Total games: {game_count}")
        print(f"  Total moves: {move_count}")
        print(f"  v7p3r moves: {v7p3r_moves}")

if __name__ == "__main__":
    create_test_data()
