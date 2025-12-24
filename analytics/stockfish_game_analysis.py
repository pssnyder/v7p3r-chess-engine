"""
Stockfish Analysis for V7P3R Bot Games
Analyzes recent losses to determine engine strength and identify patterns.
"""

import chess
import chess.pgn
import chess.engine
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Configuration
STOCKFISH_PATH = r"S:\Programming\Chess Engines\Tournament Engines\downloaded_engines\stockfish\stockfish-windows-x86-64-avx2.exe"
PGN_FILE = r"s:\Programming\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot\lichess_v7p3r_bot_2025-12-21.pgn"
OUTPUT_FILE = r"s:\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\analytics\v18_stockfish_analysis.json"
ANALYSIS_DEPTH = 20  # Depth for Stockfish analysis
GAMES_TO_ANALYZE = 20  # Total games (10 white losses, 10 black losses)

def parse_pgn_games(pgn_file: str) -> List[chess.pgn.Game]:
    """Parse all games from PGN file."""
    games = []
    with open(pgn_file, 'r', encoding='utf-8') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
    return games

def identify_losses(games: List[chess.pgn.Game], target_white: int = 10, target_black: int = 10) -> Dict[str, List[chess.pgn.Game]]:
    """Identify losses as White and Black."""
    white_losses = []
    black_losses = []
    
    for game in games:
        result = game.headers.get("Result", "*")
        white_player = game.headers.get("White", "").lower()
        black_player = game.headers.get("Black", "").lower()
        
        is_v7p3r_white = "v7p3r" in white_player
        is_v7p3r_black = "v7p3r" in black_player
        
        # Check if V7P3R lost
        if is_v7p3r_white and result == "0-1" and len(white_losses) < target_white:
            white_losses.append(game)
        elif is_v7p3r_black and result == "1-0" and len(black_losses) < target_black:
            black_losses.append(game)
        
        if len(white_losses) >= target_white and len(black_losses) >= target_black:
            break
    
    return {
        "white_losses": white_losses,
        "black_losses": black_losses
    }

def analyze_position(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> Dict[str, Any]:
    """Analyze a single position with Stockfish."""
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        
        score = info.get("score", chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))
        score_cp = score.white().score(mate_score=10000)  # Convert to centipawns
        
        best_move = info.get("pv", [None])[0]
        
        return {
            "score_cp": score_cp,
            "best_move": best_move.uci() if best_move else None,
            "depth": info.get("depth", depth),
            "is_mate": score.white().is_mate() if score.white() else False,
            "mate_in": score.white().mate() if score.white() and score.white().is_mate() else None
        }
    except Exception as e:
        print(f"Error analyzing position: {e}")
        return {
            "score_cp": 0,
            "best_move": None,
            "depth": 0,
            "is_mate": False,
            "mate_in": None,
            "error": str(e)
        }

def analyze_game(engine: chess.engine.SimpleEngine, game: chess.pgn.Game, depth: int) -> Dict[str, Any]:
    """Analyze all moves in a game."""
    board = game.board()
    move_analysis = []
    
    # Get game metadata
    white = game.headers.get("White", "Unknown")
    black = game.headers.get("Black", "Unknown")
    result = game.headers.get("Result", "*")
    opening = game.headers.get("Opening", "Unknown")
    time_control = game.headers.get("TimeControl", "Unknown")
    termination = game.headers.get("Termination", "Unknown")
    game_url = game.headers.get("Site", "Unknown")
    
    is_white = "v7p3r" in white.lower()
    
    move_num = 0
    blunders = []
    mistakes = []
    inaccuracies = []
    
    prev_score = 0
    
    for move in game.mainline_moves():
        move_num += 1
        
        # Analyze position before move
        analysis_before = analyze_position(engine, board, depth)
        
        # Make the move
        played_move = move.uci()
        board.push(move)
        
        # Analyze position after move
        analysis_after = analyze_position(engine, board, depth)
        
        # Calculate score change from V7P3R's perspective
        current_score = analysis_before["score_cp"]
        next_score = analysis_after["score_cp"]
        
        # Flip score if V7P3R is black
        if not is_white:
            current_score = -current_score
            next_score = -next_score
        
        score_loss = current_score - next_score
        
        # Categorize move quality
        move_quality = "good"
        if score_loss > 300:
            blunders.append(move_num)
            move_quality = "blunder"
        elif score_loss > 150:
            mistakes.append(move_num)
            move_quality = "mistake"
        elif score_loss > 50:
            inaccuracies.append(move_num)
            move_quality = "inaccuracy"
        
        # Only store key information to keep JSON manageable
        move_data = {
            "move_num": move_num,
            "played_move": played_move,
            "best_move": analysis_before["best_move"],
            "score_before": current_score,
            "score_after": next_score,
            "score_loss": score_loss,
            "quality": move_quality
        }
        
        move_analysis.append(move_data)
        prev_score = current_score
    
    # Calculate statistics
    total_moves = len(move_analysis)
    v7p3r_moves = total_moves // 2 if total_moves % 2 == 0 else (total_moves + 1) // 2
    
    # Calculate average centipawn loss (ACPL)
    total_loss = sum(m["score_loss"] for m in move_analysis if m["score_loss"] > 0)
    acpl = total_loss / v7p3r_moves if v7p3r_moves > 0 else 0
    
    return {
        "game_info": {
            "white": white,
            "black": black,
            "result": result,
            "opening": opening,
            "time_control": time_control,
            "termination": termination,
            "url": game_url,
            "v7p3r_color": "white" if is_white else "black"
        },
        "statistics": {
            "total_moves": total_moves,
            "v7p3r_moves": v7p3r_moves,
            "blunders": len(blunders),
            "mistakes": len(mistakes),
            "inaccuracies": len(inaccuracies),
            "acpl": round(acpl, 2),
            "blunder_moves": blunders,
            "mistake_moves": mistakes,
            "inaccuracy_moves": inaccuracies
        },
        "move_analysis": move_analysis
    }

def main():
    print("=" * 80)
    print("V7P3R Stockfish Game Analysis")
    print("=" * 80)
    
    # Parse games from PGN
    print(f"\n📂 Parsing PGN file: {PGN_FILE}")
    games = parse_pgn_games(PGN_FILE)
    print(f"   Found {len(games)} total games")
    
    # Identify losses
    print(f"\n🔍 Identifying losses...")
    losses = identify_losses(games, target_white=10, target_black=10)
    white_losses = losses["white_losses"]
    black_losses = losses["black_losses"]
    
    print(f"   White losses: {len(white_losses)}")
    print(f"   Black losses: {len(black_losses)}")
    
    # Initialize Stockfish
    print(f"\n🐟 Starting Stockfish engine...")
    print(f"   Path: {STOCKFISH_PATH}")
    print(f"   Analysis depth: {ANALYSIS_DEPTH}")
    
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    
    # Analyze games
    analysis_results = {
        "metadata": {
            "analysis_date": datetime.now().isoformat(),
            "stockfish_path": STOCKFISH_PATH,
            "analysis_depth": ANALYSIS_DEPTH,
            "pgn_source": PGN_FILE,
            "total_games_analyzed": len(white_losses) + len(black_losses)
        },
        "white_losses": [],
        "black_losses": []
    }
    
    print(f"\n📊 Analyzing games...")
    print(f"   This will take a while (depth {ANALYSIS_DEPTH} analysis)...\n")
    
    # Analyze white losses
    for i, game in enumerate(white_losses, 1):
        opponent = game.headers.get("Black", "Unknown")
        url = game.headers.get("Site", "Unknown")
        print(f"   [{i}/{len(white_losses)}] Analyzing White loss vs {opponent}...")
        print(f"            URL: {url}")
        
        result = analyze_game(engine, game, ANALYSIS_DEPTH)
        analysis_results["white_losses"].append(result)
    
    # Analyze black losses
    for i, game in enumerate(black_losses, 1):
        opponent = game.headers.get("White", "Unknown")
        url = game.headers.get("Site", "Unknown")
        print(f"   [{i}/{len(black_losses)}] Analyzing Black loss vs {opponent}...")
        print(f"            URL: {url}")
        
        result = analyze_game(engine, game, ANALYSIS_DEPTH)
        analysis_results["black_losses"].append(result)
    
    # Close engine
    engine.quit()
    
    # Save results
    print(f"\n💾 Saving analysis to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    all_games = analysis_results["white_losses"] + analysis_results["black_losses"]
    
    total_blunders = sum(g["statistics"]["blunders"] for g in all_games)
    total_mistakes = sum(g["statistics"]["mistakes"] for g in all_games)
    total_inaccuracies = sum(g["statistics"]["inaccuracies"] for g in all_games)
    
    avg_acpl = sum(g["statistics"]["acpl"] for g in all_games) / len(all_games)
    avg_blunders = total_blunders / len(all_games)
    avg_mistakes = total_mistakes / len(all_games)
    avg_inaccuracies = total_inaccuracies / len(all_games)
    
    print(f"\nGames Analyzed: {len(all_games)}")
    print(f"  - White losses: {len(analysis_results['white_losses'])}")
    print(f"  - Black losses: {len(analysis_results['black_losses'])}")
    
    print(f"\nAverage Statistics (per game):")
    print(f"  - ACPL (Average Centipawn Loss): {avg_acpl:.2f}")
    print(f"  - Blunders: {avg_blunders:.2f}")
    print(f"  - Mistakes: {avg_mistakes:.2f}")
    print(f"  - Inaccuracies: {avg_inaccuracies:.2f}")
    
    print(f"\nTotal Errors:")
    print(f"  - Blunders: {total_blunders}")
    print(f"  - Mistakes: {total_mistakes}")
    print(f"  - Inaccuracies: {total_inaccuracies}")
    
    # Identify worst games
    print(f"\nWorst Performances (by ACPL):")
    sorted_games = sorted(all_games, key=lambda x: x["statistics"]["acpl"], reverse=True)
    for i, game in enumerate(sorted_games[:5], 1):
        color = game["game_info"]["v7p3r_color"]
        opponent = game["game_info"]["black" if color == "white" else "white"]
        acpl = game["statistics"]["acpl"]
        blunders = game["statistics"]["blunders"]
        print(f"  {i}. vs {opponent} (as {color}): ACPL={acpl}, Blunders={blunders}")
        print(f"     URL: {game['game_info']['url']}")
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
