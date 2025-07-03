#!/usr/bin/env python3
"""
Script to inspect the chess_analytics.db structure
"""
import sqlite3
import os

def inspect_analytics_database():
    db_path = os.path.join("metrics", "chess_analytics.db")
    
    if not os.path.exists(db_path):
        print(f"Analytics database not found at: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in analytics database:")
    for table in tables:
        print(f"  - {table[0]}")
        
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table[0]});")
        columns = cursor.fetchall()
        print(f"    Columns in {table[0]}:")
        for col in columns:
            print(f"      {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULL'}")
        
        # Get count of records
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]};")
        count = cursor.fetchone()[0]
        print(f"    Records: {count}")
        print()
    
    conn.close()

if __name__ == "__main__":
    inspect_analytics_database()
