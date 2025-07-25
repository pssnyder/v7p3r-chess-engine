#!/usr/bin/env python3
"""
Check game and move recording issues
"""
import sqlite3

def check_games_and_moves():
    db_path = "metrics/chess_metrics_v2.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("GAME AND MOVE RECORDING CHECK")
    print("=" * 40)
    
    # Get all games
    cursor.execute("SELECT id, game_id, pgn_filename, timestamp, white_player, black_player FROM games ORDER BY timestamp DESC LIMIT 5")
    games = cursor.fetchall()
    
    print(f"Recent games ({len(games)}):")
    for game in games:
        game_id = game[1]
        print(f"  Game: {game_id} - {game[2]} ({game[3]})")
        print(f"    Players: {game[4]} vs {game[5]}")
        
        # Check moves for this game
        cursor.execute("SELECT COUNT(*) FROM move_metrics WHERE game_id = ?", (game_id,))
        move_count = cursor.fetchone()[0]
        print(f"    Moves recorded: {move_count}")
        
        if move_count > 0:
            cursor.execute("SELECT move_number, player_color, move_san, total_score FROM move_metrics WHERE game_id = ? ORDER BY move_number LIMIT 3", (game_id,))
            moves = cursor.fetchall()
            for move in moves:
                print(f"      Move {move[0]} ({move[1]}): {move[2]} - Score: {move[3]}")
        print()
    
    # Check total move count
    cursor.execute("SELECT COUNT(*) FROM move_metrics")
    total_moves = cursor.fetchone()[0]
    print(f"Total moves in database: {total_moves}")
    
    # Check for recent moves
    cursor.execute("SELECT game_id, move_number, move_san, created_at FROM move_metrics ORDER BY created_at DESC LIMIT 5")
    recent_moves = cursor.fetchall()
    
    print(f"\nMost recent move entries ({len(recent_moves)}):")
    for move in recent_moves:
        print(f"  {move[0]} - Move {move[1]}: {move[2]} ({move[3]})")
    
    conn.close()

if __name__ == "__main__":
    check_games_and_moves()
