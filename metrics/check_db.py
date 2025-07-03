import sqlite3

conn = sqlite3.connect('chess_metrics_v2.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"Tables found: {tables}")

# Check if enhanced_move_metrics table exists and get schema
for table in tables:
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    print(f"\nTable: {table}")
    print(f"Columns: {[col[1] for col in columns]}")

conn.close()
