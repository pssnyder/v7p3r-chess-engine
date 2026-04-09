"""
V7P3R Bot Overview Analysis
Quick analysis of lichess_v7p3r_bot_2026-04-09.pgn to get comprehensive stats
"""
import chess.pgn
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import json

# PGN file path (cross-workspace)
PGN_FILE = Path(r"e:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot\lichess_v7p3r_bot_2026-04-09.pgn")

def analyze_pgn_overview(pgn_file: Path):
    """Extract comprehensive stats from PGN file."""
    
    stats = {
        "total_games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "wins_white": 0,
        "wins_black": 0,
        "losses_white": 0,
        "losses_black": 0,
        "draws_white": 0,
        "draws_black": 0,
        "time_controls": Counter(),
        "opponents": Counter(),
        "opponent_ratings": [],
        "bot_ratings": [],
        "results_by_rating_bracket": defaultdict(lambda: {"W": 0, "L": 0, "D": 0}),
        "terminations": Counter(),
        "opening_families": Counter(),
        "games_by_date": Counter(),
        "time_forfeits": 0,
        "mate_wins": 0,
        "mate_losses": 0,
        "resignation_wins": 0,
        "resignation_losses": 0,
    }
    
    with open(pgn_file, 'r', encoding='utf-8') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            stats["total_games"] += 1
            
            # Extract headers
            headers = game.headers
            result = headers.get("Result", "*")
            white = headers.get("White", "")
            black = headers.get("Black", "")
            white_elo = headers.get("WhiteElo", "?")
            black_elo = headers.get("BlackElo", "?")
            time_control = headers.get("TimeControl", "unknown")
            termination = headers.get("Termination", "unknown")
            opening = headers.get("Opening", "unknown")
            date = headers.get("UTCDate", "unknown")
            
            # Determine bot color and opponent
            if "v7p3r" in white.lower():
                bot_color = "white"
                opponent = black
                opponent_elo = black_elo
                bot_elo = white_elo
            else:
                bot_color = "black"
                opponent = white
                opponent_elo = white_elo
                bot_elo = black_elo
            
            # Record stats
            stats["time_controls"][time_control] += 1
            stats["opponents"][opponent] += 1
            stats["terminations"][termination] += 1
            stats["opening_families"][opening.split(':')[0].strip()] += 1
            stats["games_by_date"][date] += 1
            
            # Convert ratings
            try:
                opp_rating = int(opponent_elo) if opponent_elo != "?" else None
                if opp_rating:
                    stats["opponent_ratings"].append(opp_rating)
            except:
                opp_rating = None
            
            try:
                b_rating = int(bot_elo) if bot_elo != "?" else None
                if b_rating:
                    stats["bot_ratings"].append(b_rating)
            except:
                b_rating = None
            
            # Results by rating bracket
            if opp_rating:
                bracket = (opp_rating // 100) * 100
                result_key = "W" if (result == "1-0" and bot_color == "white") or (result == "0-1" and bot_color == "black") else \
                             "L" if (result == "0-1" and bot_color == "white") or (result == "1-0" and bot_color == "black") else "D"
                stats["results_by_rating_bracket"][bracket][result_key] += 1
            
            # Determine win/loss/draw
            if result == "1-0":
                if bot_color == "white":
                    stats["wins"] += 1
                    stats["wins_white"] += 1
                else:
                    stats["losses"] += 1
                    stats["losses_black"] += 1
            elif result == "0-1":
                if bot_color == "black":
                    stats["wins"] += 1
                    stats["wins_black"] += 1
                else:
                    stats["losses"] += 1
                    stats["losses_white"] += 1
            elif result == "1/2-1/2":
                stats["draws"] += 1
                if bot_color == "white":
                    stats["draws_white"] += 1
                else:
                    stats["draws_black"] += 1
            
            # Termination analysis
            if "time" in termination.lower():
                stats["time_forfeits"] += 1
            
            if termination == "Normal":
                # Check if it's a checkmate
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                
                if board.is_checkmate():
                    if (board.turn == chess.BLACK and bot_color == "white") or \
                       (board.turn == chess.WHITE and bot_color == "black"):
                        stats["mate_wins"] += 1
                    else:
                        stats["mate_losses"] += 1
            
            if "abandoned" in termination.lower() or "resignation" in termination.lower():
                if result == "1-0" and bot_color == "white":
                    stats["resignation_wins"] += 1
                elif result == "0-1" and bot_color == "black":
                    stats["resignation_wins"] += 1
                elif result == "1-0" and bot_color == "black":
                    stats["resignation_losses"] += 1
                elif result == "0-1" and bot_color == "white":
                    stats["resignation_losses"] += 1
    
    return stats


def print_analysis_report(stats: dict):
    """Print formatted analysis report."""
    
    print("\n" + "="*80)
    print("V7P3R BOT PERFORMANCE OVERVIEW")
    print("="*80)
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total Games:    {stats['total_games']}")
    print(f"   Wins:           {stats['wins']} ({stats['wins']/stats['total_games']*100:.1f}%)")
    print(f"   Losses:         {stats['losses']} ({stats['losses']/stats['total_games']*100:.1f}%)")
    print(f"   Draws:          {stats['draws']} ({stats['draws']/stats['total_games']*100:.1f}%)")
    print(f"   Win Rate:       {stats['wins']/(stats['wins']+stats['losses'])*100:.1f}%")
    
    print(f"\n⚪ WHITE PERFORMANCE")
    total_white = stats['wins_white'] + stats['losses_white'] + stats['draws_white']
    if total_white > 0:
        print(f"   Games:          {total_white}")
        print(f"   Wins:           {stats['wins_white']} ({stats['wins_white']/total_white*100:.1f}%)")
        print(f"   Losses:         {stats['losses_white']} ({stats['losses_white']/total_white*100:.1f}%)")
        print(f"   Draws:          {stats['draws_white']} ({stats['draws_white']/total_white*100:.1f}%)")
    
    print(f"\n⚫ BLACK PERFORMANCE")
    total_black = stats['wins_black'] + stats['losses_black'] + stats['draws_black']
    if total_black > 0:
        print(f"   Games:          {total_black}")
        print(f"   Wins:           {stats['wins_black']} ({stats['wins_black']/total_black*100:.1f}%)")
        print(f"   Losses:         {stats['losses_black']} ({stats['losses_black']/total_black*100:.1f}%)")
        print(f"   Draws:          {stats['draws_black']} ({stats['draws_black']/total_black*100:.1f}%)")
    
    print(f"\n📈 RATING INFORMATION")
    if stats['bot_ratings']:
        avg_bot = sum(stats['bot_ratings']) / len(stats['bot_ratings'])
        print(f"   Bot Rating:     {int(avg_bot)} (avg) | {min(stats['bot_ratings'])} - {max(stats['bot_ratings'])} (range)")
    
    if stats['opponent_ratings']:
        avg_opp = sum(stats['opponent_ratings']) / len(stats['opponent_ratings'])
        print(f"   Opponent Avg:   {int(avg_opp)} | {min(stats['opponent_ratings'])} - {max(stats['opponent_ratings'])} (range)")
    
    print(f"\n📊 RESULTS BY OPPONENT RATING")
    for bracket in sorted(stats['results_by_rating_bracket'].keys()):
        data = stats['results_by_rating_bracket'][bracket]
        total = data['W'] + data['L'] + data['D']
        win_pct = data['W'] / total * 100 if total > 0 else 0
        print(f"   {bracket}-{bracket+99}: {data['W']}W {data['L']}L {data['D']}D ({win_pct:.0f}% win rate, {total} games)")
    
    print(f"\n⏱️  TIME CONTROLS")
    for tc, count in stats['time_controls'].most_common(10):
        print(f"   {tc:<20} {count} games ({count/stats['total_games']*100:.1f}%)")
    
    print(f"\n🎯 GAME TERMINATIONS")
    print(f"   Checkmate Wins:      {stats['mate_wins']}")
    print(f"   Checkmate Losses:    {stats['mate_losses']}")
    print(f"   Resignation Wins:    {stats['resignation_wins']}")
    print(f"   Resignation Losses:  {stats['resignation_losses']}")
    print(f"   Time Forfeits:       {stats['time_forfeits']}")
    
    print(f"\n🏆 TOP OPPONENTS")
    for opponent, count in stats['opponents'].most_common(10):
        print(f"   {opponent:<25} {count} games")
    
    print(f"\n📅 ACTIVITY BY DATE")
    for date in sorted(stats['games_by_date'].keys())[-10:]:  # Last 10 dates
        count = stats['games_by_date'][date]
        print(f"   {date}: {count} games")
    
    print(f"\n♟️  TOP OPENINGS")
    for opening, count in stats['opening_families'].most_common(10):
        print(f"   {opening:<35} {count} games ({count/stats['total_games']*100:.1f}%)")
    
    print("\n" + "="*80)
    

if __name__ == "__main__":
    print("Analyzing V7P3R Bot game history...")
    print(f"Reading from: {PGN_FILE}\n")
    
    if not PGN_FILE.exists():
        print(f"ERROR: PGN file not found at {PGN_FILE}")
        exit(1)
    
    stats = analyze_pgn_overview(PGN_FILE)
    print_analysis_report(stats)
    
    # Save to JSON
    output_file = Path(__file__).parent / "bot_overview_stats.json"
    
    # Convert Counter objects to regular dicts for JSON serialization
    json_stats = {
        k: dict(v) if isinstance(v, Counter) else 
           {str(kk): vv for kk, vv in v.items()} if isinstance(v, defaultdict) else v
        for k, v in stats.items()
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    print(f"\n💾 Detailed stats saved to: {output_file}")
