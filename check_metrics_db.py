#!/usr/bin/env python3
"""
Diagnostic script to check what's actually in the metrics database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metrics.metrics_store import MetricsStore

def check_database_content():
    """Check what's actually in the database"""
    print("=== Metrics Database Diagnostic ===")
    
    metrics_store = MetricsStore()
    
    # Check game results
    connection = metrics_store._get_connection()
    with connection:
        cursor = connection.cursor()
        
        # Check table structure
        print("\n1. Game Results Table Structure:")
        cursor.execute("PRAGMA table_info(game_results)")
        columns = cursor.fetchall()
        for col in columns:
            print(f"   {col[1]} ({col[2]})")
        
        # Check game results data
        print("\n2. Game Results Sample:")
        cursor.execute("SELECT game_id, white_player, black_player, winner FROM game_results LIMIT 5")
        games = cursor.fetchall()
        for game in games:
            print(f"   Game: {game[0]}, White: {game[1]}, Black: {game[2]}, Winner: {game[3]}")
        
        print(f"\n   Total games: {len(cursor.execute('SELECT * FROM game_results').fetchall())}")
        
        # Check move metrics
        print("\n3. Move Metrics Table Structure:")
        cursor.execute("PRAGMA table_info(move_metrics)")
        columns = cursor.fetchall()
        for col in columns:
            print(f"   {col[1]} ({col[2]})")
        
        print("\n4. Move Metrics Sample:")
        cursor.execute("SELECT game_id, player_color, move_uci, engine_type FROM move_metrics LIMIT 5")
        moves = cursor.fetchall()
        for move in moves:
            print(f"   Game: {move[0]}, Color: {move[1]}, Move: {move[2]}, Engine: {move[3]}")
        
        print(f"\n   Total moves: {len(cursor.execute('SELECT * FROM move_metrics').fetchall())}")
        
        # Check unique engine types
        print("\n5. Unique Engine Types in Move Metrics:")
        cursor.execute("SELECT DISTINCT engine_type FROM move_metrics")
        engine_types = cursor.fetchall()
        for engine_type in engine_types:
            print(f"   {engine_type[0]}")
        
        # Check unique players
        print("\n6. Unique Players:")
        cursor.execute("SELECT DISTINCT white_player FROM game_results UNION SELECT DISTINCT black_player FROM game_results")
        players = cursor.fetchall()
        for player in players:
            print(f"   {player[0]}")
        
        # Check for V7P3R specific data
        print("\n7. V7P3R Related Data:")
        cursor.execute("SELECT COUNT(*) FROM move_metrics WHERE engine_type LIKE '%v7p3r%' OR engine_type LIKE '%V7P3R%'")
        v7p3r_moves = cursor.fetchone()[0]
        print(f"   V7P3R moves: {v7p3r_moves}")
        
        cursor.execute("SELECT COUNT(*) FROM game_results WHERE white_player LIKE '%V7P3R%' OR black_player LIKE '%V7P3R%'")
        v7p3r_games = cursor.fetchone()[0]
        print(f"   V7P3R games: {v7p3r_games}")

if __name__ == "__main__":
    check_database_content()
