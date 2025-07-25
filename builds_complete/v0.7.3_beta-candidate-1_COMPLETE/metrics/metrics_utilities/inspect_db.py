#!/usr/bin/env python3
"""
Script to inspect the chess_metrics.db to understand the data being stored
"""
import sqlite3
import os

def inspect_database():
    db_path = os.path.join("metrics", "chess_metrics.db")
    
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in database:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # Check the move_metrics table specifically
    if any(table[0] == 'move_metrics' for table in tables):
        print("\n=== MOVE_METRICS TABLE ===")
        
        # Get table schema
        cursor.execute("PRAGMA table_info(move_metrics);")
        columns = cursor.fetchall()
        print("Columns:")
        for col in columns:
            print(f"  {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULL'} - {'DEFAULT: ' + str(col[4]) if col[4] else ''}")
        
        # Get count of records
        cursor.execute("SELECT COUNT(*) FROM move_metrics;")
        count = cursor.fetchone()[0]
        print(f"\nTotal records: {count}")
        
        # Get sample data with engine information
        print("\nSample data with engine details (last 10 records):")
        cursor.execute("""
            SELECT game_id, move_number, move_uci, time_taken, nodes_searched, depth, 
                   engine_name, search_algorithm, pv_line 
            FROM move_metrics 
            ORDER BY rowid DESC 
            LIMIT 10
        """)
        rows = cursor.fetchall()
        
        for row in rows:
            print(f"Game: {row[0]}, Move: {row[1]}, UCI: {row[2]}")
            print(f"  Time: {row[3]}, Nodes: {row[4]}, Depth: {row[5]}")
            print(f"  Engine: {row[6]}, Algorithm: {row[7]}")
            print(f"  PV: {row[8]}")
            print()
        
        # Get statistics on the data
        print("=== DATA STATISTICS ===")
        cursor.execute("""
            SELECT 
                COUNT(*) as total_moves,
                COUNT(CASE WHEN time_taken > 0 THEN 1 END) as moves_with_time,
                COUNT(CASE WHEN nodes_searched > 0 THEN 1 END) as moves_with_nodes,
                COUNT(CASE WHEN depth > 0 THEN 1 END) as moves_with_depth,
                AVG(CASE WHEN time_taken > 0 THEN time_taken END) as avg_time,
                AVG(CASE WHEN nodes_searched > 0 THEN nodes_searched END) as avg_nodes
            FROM move_metrics
        """)
        stats = cursor.fetchone()
        print(f"Total moves: {stats[0]}")
        print(f"Moves with time > 0: {stats[1]}")
        print(f"Moves with nodes > 0: {stats[2]}")
        print(f"Moves with depth > 0: {stats[3]}")
        print(f"Average time (for moves > 0): {stats[4]:.2f}" if stats[4] else "No valid time data")
        print(f"Average nodes (for moves > 0): {stats[5]:.0f}" if stats[5] else "No valid nodes data")
    
    conn.close()

if __name__ == "__main__":
    inspect_database()
