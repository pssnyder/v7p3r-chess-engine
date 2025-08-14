
import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'engine_metrics.db')

def get_latest_game_id(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM game_results ORDER BY timestamp DESC LIMIT 1')
    row = cursor.fetchone()
    return row[0] if row else None

def print_move_analysis_for_game(game_id, conn):
    cursor = conn.cursor()
    cursor.execute('''
        SELECT move_number, player_color, move_uci, evaluation_score, search_depth, nodes_searched, search_time, evaluation_details
        FROM move_analysis
        WHERE game_id = ?
        ORDER BY move_number ASC
    ''', (game_id,))
    rows = cursor.fetchall()
    print(f"Move analysis for game_id {game_id}:")
    for row in rows:
        move_number, player_color, move_uci, eval_score, depth, nodes, time_spent, eval_details = row
        eval_score_str = f"{eval_score:8.2f}" if eval_score is not None else "   N/A  "
        depth_str = str(depth) if depth is not None else "N/A"
        nodes_str = str(nodes) if nodes is not None else "N/A"
        time_str = f"{time_spent:.2f}s" if time_spent is not None else "N/A"
        print(f"Move {move_number:2d} | {player_color:5s} | {move_uci:6s} | Eval: {eval_score_str} | Depth: {depth_str} | Nodes: {nodes_str} | Time: {time_str}")
        if eval_details:
            try:
                details = json.loads(eval_details)
                print(f"    Details: {json.dumps(details, indent=4)}")
            except Exception:
                print(f"    Details: {eval_details}")

def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        game_id = get_latest_game_id(conn)
        if not game_id:
            print("No games found in the database.")
            return
        print_move_analysis_for_game(game_id, conn)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
