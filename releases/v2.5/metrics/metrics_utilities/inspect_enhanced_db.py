#!/usr/bin/env python3
"""
Inspect the enhanced metrics database to see the new detailed data
"""
import sqlite3
import json

def inspect_enhanced_database():
    db_path = "metrics/chess_metrics_v2.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("ENHANCED CHESS METRICS DATABASE INSPECTION")
    print("=" * 50)
    
    # Show table structure
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables: {[t[0] for t in tables]}")
    print()
    
    # Check games
    cursor.execute("SELECT COUNT(*) FROM games")
    game_count = cursor.fetchone()[0]
    print(f"Total games: {game_count}")
    
    if game_count > 0:
        cursor.execute("SELECT * FROM games ORDER BY timestamp DESC LIMIT 1")
        latest_game = cursor.fetchone()
        print(f"Latest game: {latest_game[1]} ({latest_game[3]} vs {latest_game[4]})")
        print()
    
    # Check engine configurations
    cursor.execute("SELECT COUNT(*) FROM engine_configs")
    config_count = cursor.fetchone()[0]
    print(f"Engine configurations: {config_count}")
    
    if config_count > 0:
        cursor.execute("SELECT engine_name, engine_version, search_algorithm FROM engine_configs")
        configs = cursor.fetchall()
        for config in configs:
            print(f"  {config[0]} v{config[1]} ({config[2]})")
        print()
    
    # Check move metrics with detailed scoring
    cursor.execute("SELECT COUNT(*) FROM move_metrics")
    move_count = cursor.fetchone()[0]
    print(f"Total moves recorded: {move_count}")
    
    if move_count > 0:
        print("\nSample enhanced move data:")
        cursor.execute("""
            SELECT 
                game_id, move_number, player_color, move_san, move_uci,
                time_taken, nodes_searched, evaluation, game_phase,
                total_score, material_score, king_safety_score, center_control_score,
                pawn_structure_score, mobility_score
            FROM move_metrics 
            ORDER BY created_at DESC 
            LIMIT 3
        """)
        
        moves = cursor.fetchall()
        for move in moves:
            print(f"  Move {move[1]} ({move[2]}): {move[3]} [{move[4]}]")
            print(f"    Time: {move[5]:.4f}s, Nodes: {move[6]}, Eval: {move[7]:.2f}")
            print(f"    Phase: {move[8]}")
            print(f"    Detailed Scores:")
            print(f"      Total: {move[9] or 0:.2f}, Material: {move[10] or 0:.2f}")
            print(f"      King Safety: {move[11] or 0:.2f}, Center Control: {move[12] or 0:.2f}")
            print(f"      Pawn Structure: {move[13] or 0:.2f}, Mobility: {move[14] or 0:.2f}")
            print()
    
    # Check search efficiency data
    cursor.execute("SELECT COUNT(*) FROM search_efficiency")
    search_count = cursor.fetchone()[0]
    print(f"Search efficiency records: {search_count}")
    
    # Show performance view
    cursor.execute("SELECT * FROM engine_performance")
    performance = cursor.fetchall()
    if performance:
        print("\nEngine Performance Summary:")
        for perf in performance:
            print(f"  {perf[0]} v{perf[1]} ({perf[2]}):")
            avg_time = perf[4] if perf[4] is not None else 0.0
            avg_nodes = perf[5] if perf[5] is not None else 0
            avg_eval = perf[6] if perf[6] is not None else 0.0
            wins = perf[7] if perf[7] is not None else 0
            draws = perf[8] if perf[8] is not None else 0
            losses = perf[9] if perf[9] is not None else 0
            print(f"    Games: {perf[3]}, Avg Time: {avg_time:.3f}s")
            print(f"    Avg Nodes: {avg_nodes:.0f}, Avg Eval: {avg_eval:.2f}")
            print(f"    W-D-L: {wins}-{draws}-{losses}")
            print()
    
    conn.close()

if __name__ == "__main__":
    inspect_enhanced_database()
