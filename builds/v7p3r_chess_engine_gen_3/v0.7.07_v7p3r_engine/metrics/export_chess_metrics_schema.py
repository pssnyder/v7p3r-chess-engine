import sqlite3

conn = sqlite3.connect("metrics/chess_metrics.db")
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for table in tables:
    print(f"--- Schema for table: {table[0]} ---")
    cursor.execute(f"PRAGMA table_info({table[0]})")
    for col in cursor.fetchall():
        print(col)
    print()

conn.close()
