# metrics/quick_metrics.py
""" Quick Metrics Module
This module provides a quick view into recent gameplay metrics based on whatever pgn files are in the games directory.
Display:
    - Recent Game Outcomes, grouped by player
    - Win/Loss/Draw Ratios
    - Average Game Length
    - Evaluation Metrics
    - Opening Moves Statistics
"""
import os
import json
import logging
import chess.pgn

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', 'games')

logging.basicConfig(
    filename='logging/quick_metrics.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def get_pgn_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pgn')]

def parse_games(pgn_files, max_games=1000):
    games = []
    for pgn_file in sorted(pgn_files, reverse=True):
        with open(pgn_file, 'r', encoding='utf-8') as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                games.append(game)
                if len(games) >= max_games:
                    return games
    return games

def get_game_outcomes(games):
    outcomes = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
    by_player = {}
    for game in games:
        result = game.headers.get('Result', '')
        white = game.headers.get('White', 'Unknown')
        black = game.headers.get('Black', 'Unknown')
        outcomes[result] = outcomes.get(result, 0) + 1
        for player in [white, black]:
            if player not in by_player:
                by_player[player] = {'win': 0, 'loss': 0, 'draw': 0, 'games': 0}
        if result == '1-0':
            by_player[white]['win'] += 1
            by_player[black]['loss'] += 1
        elif result == '0-1':
            by_player[black]['win'] += 1
            by_player[white]['loss'] += 1
        elif result == '1/2-1/2':
            by_player[white]['draw'] += 1
            by_player[black]['draw'] += 1
        by_player[white]['games'] += 1
        by_player[black]['games'] += 1
    return outcomes, by_player

def get_average_game_length(games):
    total_moves = 0
    for game in games:
        total_moves += len(list(game.mainline_moves()))
    return total_moves / len(games) if games else 0

def get_opening_stats(games, num_moves=4):
    opening_counts = {}
    for game in games:
        board = game.board()
        moves = []
        for i, move in enumerate(game.mainline_moves()):
            if i >= num_moves:
                break
            moves.append(board.san(move))
            board.push(move)
        opening = ' '.join(moves)
        if opening:
            opening_counts[opening] = opening_counts.get(opening, 0) + 1
    return dict(sorted(opening_counts.items(), key=lambda x: x[1], reverse=True)[:10])

def get_eval_metrics(games):
    import re
    evals = []
    for game in games:
        node = game
        while node.variations:
            node = node.variation(0)
            comment = node.comment
            if comment:
                # Look for 'Eval: <value>' in the comment
                match = re.search(r'Eval:\s*([-+]?\d*\.?\d+)', comment)
                if match:
                    try:
                        evals.append(float(match.group(1)))
                    except Exception:
                        pass
        # Also check for 'Eval' in headers (legacy support)
        if 'Eval' in game.headers:
            try:
                evals.append(float(game.headers['Eval']))
            except Exception:
                pass
    if not evals:
        return {'avg_eval': None, 'min_eval': None, 'max_eval': None, 'count': 0}
    return {
        'avg_eval': sum(evals) / len(evals),
        'min_eval': min(evals),
        'max_eval': max(evals),
        'count': len(evals)
    }

def print_metrics():
    print("\n--- Quick Metrics Report ---\n")
    pgn_files = get_pgn_files(GAMES_DIR)
    if not pgn_files:
        print("No PGN files found in games directory.")
        return
    # Always use the most recent 100 games
    games = parse_games(pgn_files, max_games=1000)
    print(f"Parsed {len(games)} recent games.")

    outcomes, by_player = get_game_outcomes(games)
    print("\nGame Outcomes:")
    for result, count in outcomes.items():
        print(f"  {result}: {count}")

    print("\nWin/Loss/Draw Ratios by Player:")
    for player, stats in by_player.items():
        print(f"  {player}: {stats['win']}W/{stats['loss']}L/{stats['draw']}D (Games: {stats['games']})")

    avg_length = get_average_game_length(games)
    print(f"\nAverage Game Length: {avg_length:.2f} moves")

    eval_metrics = get_eval_metrics(games)
    print("\nEvaluation Metrics:")
    if eval_metrics['count'] == 0:
        print("  No evaluation data found in recent games.")
    else:
        print(f"  Avg Eval: {eval_metrics['avg_eval']}")
        print(f"  Min Eval: {eval_metrics['min_eval']}")
        print(f"  Max Eval: {eval_metrics['max_eval']}")
        print(f"  Eval Count: {eval_metrics['count']}")

    opening_stats = get_opening_stats(games)
    print("\nTop 10 Opening Sequences:")
    for opening, count in opening_stats.items():
        print(f"  {opening}: {count}")

if __name__ == "__main__":
    print_metrics()
