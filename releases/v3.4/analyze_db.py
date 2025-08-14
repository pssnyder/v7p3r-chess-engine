import os
import sqlite3
import chess
import chess.pgn
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def copy_db_file():
    """Make a backup copy of the database file for safety"""
    src_path = "engine_metrics.db"
    if os.path.exists(src_path):
        backup_path = f"engine_metrics_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        import shutil
        shutil.copy2(src_path, backup_path)
        print(f"Created backup at {backup_path}")
    else:
        print(f"Warning: Database file {src_path} not found")

def analyze_database():
    """Analyze the database structure and extract metrics"""
    db_path = "engine_metrics.db"
    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found")
        return
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get the list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables in the database: {[t[0] for t in tables]}")
    
    # Analyze game results
    cursor.execute("SELECT id, timestamp, result, white_player, black_player, total_moves, game_duration FROM game_results")
    games = cursor.fetchall()
    print(f"\nFound {len(games)} games in the database")
    
    if games:
        print("\nGame results:")
        for i, (id, timestamp, result, white, black, moves, duration) in enumerate(games[:5]):  # Show only first 5
            print(f"Game {id}: {white} vs {black}, Result: {result}, Moves: {moves}, Duration: {duration:.2f}s")
        
        # Count results
        wins = sum(1 for _, _, res, _, _, _, _ in games if res == '1-0' or res == '0-1')
        losses = sum(1 for _, _, res, _, _, _, _ in games if res == '1-0' or res == '0-1')
        draws = sum(1 for _, _, res, _, _, _, _ in games if res == '1/2-1/2')
        in_progress = sum(1 for _, _, res, _, _, _, _ in games if res == 'In Progress')
        
        print(f"\nResults summary: {wins} wins, {losses} losses, {draws} draws, {in_progress} in progress")
    
    # Analyze move analysis
    cursor.execute("SELECT COUNT(*) FROM move_analysis")
    move_count = cursor.fetchone()[0]
    print(f"\nFound {move_count} move analyses in the database")
    
    if move_count > 0:
        cursor.execute("SELECT game_id, move_number, player_color, move_uci, evaluation_score, search_time FROM move_analysis LIMIT 10")
        moves = cursor.fetchall()
        
        print("\nSample move analyses:")
        for game_id, move_number, color, move, eval, time in moves:
            print(f"Game {game_id}, Move {move_number} ({color}): {move}, Eval: {eval}, Time: {time:.3f}s")
        
        # Get evaluation distribution
        cursor.execute("SELECT evaluation_score FROM move_analysis WHERE evaluation_score IS NOT NULL")
        evals = [row[0] for row in cursor.fetchall() if row[0] is not None]
        
        if evals:
            print(f"\nEvaluation statistics:")
            print(f"  Min: {min(evals):.2f}")
            print(f"  Max: {max(evals):.2f}")
            print(f"  Avg: {sum(evals)/len(evals):.2f}")
            print(f"  Count: {len(evals)}")
            
            # Plot evaluation histogram
            plt.figure(figsize=(10, 6))
            plt.hist(evals, bins=20, alpha=0.7)
            plt.title("V7P3R Evaluation Distribution")
            plt.xlabel("Evaluation Score")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            
            # Ensure directory exists
            if not os.path.exists('analysis_results'):
                os.makedirs('analysis_results')
                
            plt.savefig('analysis_results/evaluation_histogram.png')
            plt.close()
            print(f"Saved evaluation histogram to analysis_results/evaluation_histogram.png")
    
    # Close the connection
    conn.close()
    print("\nDatabase analysis complete")

def export_pgn_from_database():
    """Export PGN from the database"""
    db_path = "engine_metrics.db"
    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found")
        return
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get games with PGN data
    cursor.execute("SELECT id, pgn FROM game_results WHERE pgn IS NOT NULL")
    pgn_games = cursor.fetchall()
    
    if not pgn_games:
        print("No PGN data found in the database")
        conn.close()
        return
    
    print(f"Found {len(pgn_games)} games with PGN data")
    
    # Ensure directory exists
    export_dir = 'exported_pgn'
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    # Export each game
    for game_id, pgn_text in pgn_games:
        if pgn_text:
            file_path = os.path.join(export_dir, f"game_{game_id}.pgn")
            with open(file_path, 'w') as f:
                f.write(pgn_text)
            print(f"Exported game {game_id} to {file_path}")
    
    conn.close()
    print(f"Exported {len(pgn_games)} PGN files to {export_dir} directory")

if __name__ == "__main__":
    # Make a backup copy for safety
    copy_db_file()
    
    # Analyze the database
    analyze_database()
    
    # Export PGN data
    export_pgn_from_database()
