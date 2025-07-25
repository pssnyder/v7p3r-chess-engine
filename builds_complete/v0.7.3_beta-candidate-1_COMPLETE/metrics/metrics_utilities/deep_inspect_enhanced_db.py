#!/usr/bin/env python3
"""
Deep inspection of the enhanced metrics database
"""
import sqlite3
import json

def deep_inspect_enhanced_database():
    db_path = "metrics/chess_metrics_v2.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("DEEP ENHANCED DATABASE INSPECTION")
    print("=" * 50)
    
    # Get detailed move metrics info
    cursor.execute("SELECT COUNT(*) FROM move_metrics")
    move_count = cursor.fetchone()[0]
    print(f"Total move metrics: {move_count}")
    
    if move_count > 0:
        cursor.execute("""
            SELECT m.*, g.pgn_filename, ec.engine_name 
            FROM move_metrics m
            JOIN games g ON m.game_id = g.game_id
            LEFT JOIN engine_configs ec ON m.engine_config_id = ec.config_id
            ORDER BY m.move_number
        """)
        moves = cursor.fetchall()
        
        print(f"\nAll {len(moves)} move records:")
        for i, move in enumerate(moves):
            print(f"  Move {i+1}: {move[11]} - {move[12]} (game: {move[-2]})")
            print(f"    Engine: {move[-1]}, Time: {move[7]}s, Nodes: {move[8]}")
            print(f"    Detailed scores: total={move[15]}, material={move[16]}")
            print()
    
    # Check recent games
    cursor.execute("""
        SELECT g.id, g.pgn_filename, g.timestamp, 
               COUNT(m.id) as move_count
        FROM games g
        LEFT JOIN move_metrics m ON g.game_id = m.game_id
        GROUP BY g.id
        ORDER BY g.timestamp DESC
        LIMIT 5
    """)
    recent_games = cursor.fetchall()
    
    print("Recent games with move counts:")
    for game in recent_games:
        print(f"  {game[1]} ({game[2]}): {game[3]} moves")
    
    print()
    
    # Check if there are any position analysis records
    cursor.execute("SELECT COUNT(*) FROM position_analysis")
    position_count = cursor.fetchone()[0]
    print(f"Position analysis records: {position_count}")
    
    if position_count > 0:
        cursor.execute("""
            SELECT game_phase, position_type, material_balance
            FROM position_analysis
            ORDER BY id DESC
            LIMIT 5
        """)
        positions = cursor.fetchall()
        print("Recent position analyses:")
        for pos in positions:
            print(f"  Phase: {pos[0]}, Type: {pos[1]}, Balance: {pos[2]}")
    
    conn.close()

if __name__ == "__main__":
    deep_inspect_enhanced_database()
