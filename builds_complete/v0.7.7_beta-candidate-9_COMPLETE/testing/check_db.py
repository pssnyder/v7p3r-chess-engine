#!/usr/bin/env python3
"""Simple script to check the metrics database contents."""

import sqlite3
import os

db_path = 'metrics/chess_metrics.db'

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    exit(1)

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables in database: {[t[0] for t in tables]}")
    
    # Check schema for games table
    cursor.execute("PRAGMA table_info(games)")
    games_schema = cursor.fetchall()
    print(f"\nGames table schema:")
    for column in games_schema:
        print(f"  {column[1]} ({column[2]})")
    
    # Check schema for moves table
    cursor.execute("PRAGMA table_info(moves)")
    moves_schema = cursor.fetchall()
    print(f"\nMoves table schema:")
    for column in moves_schema:
        print(f"  {column[1]} ({column[2]})")
    
    # Check games count
    cursor.execute("SELECT COUNT(*) FROM games")
    game_count = cursor.fetchone()[0]
    print(f"\nGames recorded: {game_count}")
    
    # Check moves count
    cursor.execute("SELECT COUNT(*) FROM moves")
    move_count = cursor.fetchone()[0]
    print(f"Moves recorded: {move_count}")
    
    # Show recent games if any
    if game_count > 0:
        cursor.execute("SELECT game_id, v7p3r_color, opponent, result, timestamp FROM games ORDER BY timestamp DESC LIMIT 3")
        recent_games = cursor.fetchall()
        print(f"\nRecent games:")
        for game in recent_games:
            print(f"  Game ID: {game[0]}, v7p3r: {game[1]}, Opponent: {game[2]}, Result: {game[3]}, Started: {game[4]}")
    
    # Show recent moves if any
    if move_count > 0:
        cursor.execute("SELECT game_id, move_number, player, move_notation, evaluation_score FROM moves ORDER BY id DESC LIMIT 5")
        recent_moves = cursor.fetchall()
        print(f"\nRecent moves:")
        for move in recent_moves:
            print(f"  Game: {move[0]}, Move #{move[1]}, Player: {move[2]}, Move: {move[3]}, Eval: {move[4]}")
    
    conn.close()
    print("\nDatabase check complete!")
    
except Exception as e:
    print(f"Error checking database: {e}")
