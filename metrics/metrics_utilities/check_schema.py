#!/usr/bin/env python3
"""
Check the enhanced database schema
"""
import sqlite3

def check_schema():
    db_path = "metrics/chess_metrics_v2.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get schema for each table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    
    for table in tables:
        print(f"\nTable: {table}")
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
    
    conn.close()

if __name__ == "__main__":
    check_schema()
