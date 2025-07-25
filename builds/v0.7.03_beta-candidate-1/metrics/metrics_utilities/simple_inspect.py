#!/usr/bin/env python3
"""
Simple inspection of the move metrics table
"""
import sqlite3

def simple_inspect():
    db_path = "metrics/chess_metrics_v2.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("SIMPLE MOVE METRICS INSPECTION")
    print("=" * 40)
    
    # Get all move metrics
    cursor.execute("SELECT * FROM move_metrics")
    moves = cursor.fetchall()
    
    print(f"Total moves: {len(moves)}")
    
    if moves:
        print("\nFirst move record:")
        cursor.execute("PRAGMA table_info(move_metrics)")
        columns = [col[1] for col in cursor.fetchall()]
        
        move = moves[0]
        for i, col_name in enumerate(columns):
            if i < len(move):
                print(f"  {col_name}: {move[i]}")
    
    # Check games table
    cursor.execute("SELECT id, game_id, pgn_filename, timestamp FROM games ORDER BY timestamp DESC LIMIT 3")
    games = cursor.fetchall()
    
    print(f"\nRecent games:")
    for game in games:
        print(f"  {game[1]} - {game[2]} ({game[3]})")
    
    conn.close()

if __name__ == "__main__":
    simple_inspect()
