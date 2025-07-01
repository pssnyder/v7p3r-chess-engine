#!/usr/bin/env python3
"""
Test script to validate the new metrics collection logic
"""
import sys
import os
import sqlite3
import time

sys.path.insert(0, os.path.abspath('.'))

def test_metrics_collection():
    print("Testing the updated metrics collection logic...")
    
    # Check the current database before any new game
    db_path = os.path.join("metrics", "chess_metrics.db")
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get the count of records before test
        cursor.execute("SELECT COUNT(*) FROM move_metrics")
        before_count = cursor.fetchone()[0]
        
        # Get the latest game_id
        cursor.execute("SELECT game_id FROM move_metrics ORDER BY rowid DESC LIMIT 1")
        latest_result = cursor.fetchone()
        latest_game = latest_result[0] if latest_result else "none"
        
        conn.close()
        
        print(f"Current database has {before_count} move records")
        print(f"Latest game: {latest_game}")
        
        # Look for records where nodes_searched is 0 but should have data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("\nAnalyzing recent data quality...")
        # Check all engines in the latest game
        cursor.execute("""
            SELECT engine_name, search_algorithm, player_color,
                   COUNT(*) as total_moves,
                   COUNT(CASE WHEN nodes_searched > 0 THEN 1 END) as moves_with_nodes,
                   COUNT(CASE WHEN time_taken > 0 THEN 1 END) as moves_with_time,
                   AVG(CASE WHEN nodes_searched > 0 THEN nodes_searched END) as avg_nodes,
                   AVG(time_taken) as avg_time
            FROM move_metrics 
            WHERE game_id = ?
            GROUP BY engine_name, search_algorithm, player_color
            ORDER BY player_color, engine_name
        """, (latest_game,))
        
        stats = cursor.fetchall()
        for stat in stats:
            color = "White" if stat[2] == 'w' else "Black"
            print(f"{color} - Engine: {stat[0]} ({stat[1]})")
            print(f"  Total moves: {stat[3]}")
            print(f"  Moves with nodes: {stat[4]} ({100*stat[4]/stat[3]:.1f}%)")
            print(f"  Moves with time: {stat[5]} ({100*stat[5]/stat[3]:.1f}%)")
            print(f"  Avg nodes (when > 0): {stat[6]:.0f}" if stat[6] else "  No node data")
            print(f"  Avg time: {stat[7]:.2f}s")
            print()
            
        # Also check overall statistics
        print("=== OVERALL DATABASE STATISTICS ===")
        cursor.execute("""
            SELECT engine_name, search_algorithm,
                   COUNT(*) as total_moves,
                   COUNT(CASE WHEN nodes_searched > 0 THEN 1 END) as moves_with_nodes,
                   AVG(CASE WHEN nodes_searched > 0 THEN nodes_searched END) as avg_nodes
            FROM move_metrics 
            GROUP BY engine_name, search_algorithm
            ORDER BY total_moves DESC
        """)
        
        all_stats = cursor.fetchall()
        for stat in all_stats:
            print(f"Engine: {stat[0]} ({stat[1]})")
            print(f"  Total moves: {stat[2]}")
            print(f"  Moves with nodes: {stat[3]} ({100*stat[3]/stat[2]:.1f}%)")
            print(f"  Avg nodes (when > 0): {stat[4]:.0f}" if stat[4] else "  No node data")
            print()
        
        conn.close()
    else:
        print("No database found")

if __name__ == "__main__":
    test_metrics_collection()
