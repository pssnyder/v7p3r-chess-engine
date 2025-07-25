import sqlite3

conn = sqlite3.connect('chess_analytics.db')
cursor = conn.cursor()

# Find v7p3r games
cursor.execute("SELECT game_id, white_player, black_player FROM game_results WHERE white_player = 'v7p3r' OR black_player = 'v7p3r' LIMIT 5")
v7p3r_games = cursor.fetchall()
print('V7P3R Games (first 5):')
for game in v7p3r_games:
    print(f'  {game}')

if v7p3r_games:
    game_id = v7p3r_games[0][0]
    print(f'\nChecking move metrics for game: {game_id}')
    
    # Check if there are move metrics for this game
    cursor.execute('SELECT COUNT(*) FROM move_metrics WHERE game_id = ?', (game_id,))
    count = cursor.fetchone()[0]
    print(f'Move metrics count for {game_id}: {count}')
    
    # Check with .pgn extension
    cursor.execute('SELECT COUNT(*) FROM move_metrics WHERE game_id = ?', (game_id + '.pgn',))
    count_pgn = cursor.fetchone()[0]
    print(f'Move metrics count for {game_id}.pgn: {count_pgn}')
    
    # Test the problematic query from the dashboard
    print(f'\nTesting dashboard query logic:')
    v7p3r_game_ids = [g[0] for g in v7p3r_games]
    extended_game_ids = v7p3r_game_ids + [gid + '.pgn' for gid in v7p3r_game_ids]
    print(f'Extended game IDs: {extended_game_ids[:5]}...')
    
    # Test the JOIN query specifically for evaluation metric
    selected_metric = 'evaluation'
    extended_placeholders = ','.join(['?'] * len(extended_game_ids))
    query = f"""
    SELECT mm.*, gr.white_player, gr.black_player, gr.winner
    FROM move_metrics mm
    JOIN game_results gr ON (mm.game_id = gr.game_id OR mm.game_id = gr.game_id || '.pgn')
    WHERE mm.game_id IN ({extended_placeholders})
    AND mm.{selected_metric} IS NOT NULL
    ORDER BY mm.created_at
    LIMIT 5
    """
    
    try:
        cursor.execute(query, extended_game_ids)
        results = cursor.fetchall()
        print(f'Query results count: {len(results)}')
        if results:
            print('Sample results:')
            for i, row in enumerate(results[:3]):
                print(f'  {i+1}: game_id={row[1]}, move_number={row[2]}, player_color={row[3]}, evaluation={row[6]}')
        else:
            print('No results from JOIN query')
            
            # Try a simpler query
            cursor.execute(f"SELECT COUNT(*) FROM move_metrics WHERE game_id IN ({extended_placeholders}) AND {selected_metric} IS NOT NULL", extended_game_ids)
            simple_count = cursor.fetchone()[0]
            print(f'Simple count without JOIN: {simple_count}')
            
    except Exception as e:
        print(f'Error executing query: {e}')

conn.close()
