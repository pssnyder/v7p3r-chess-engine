import sqlite3

# Connect to the database
conn = sqlite3.connect('engine_metrics.db')
cursor = conn.cursor()

# Get list of tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables in the database:")
for table in tables:
    print(f"- {table[0]}")

# For each table, get column names
for table_name in [t[0] for t in tables]:
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f"\nColumns in table '{table_name}':")
    for col in columns:
        print(f"- {col[1]} ({col[2]})")
    
    # Print a few rows from each table
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
    rows = cursor.fetchall()
    print(f"\nSample data from '{table_name}' ({len(rows)} rows):")
    for row in rows:
        print(row)

# Close the connection
conn.close()
