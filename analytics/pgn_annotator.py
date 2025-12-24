"""
PGN Game Annotator with Stockfish
Analyzes chess games and produces annotated PGN with evaluations, best moves, and mistake classifications.
Similar to Lichess/Chess.com post-game analysis.

Usage: python pgn_annotator.py [input.pgn] [output.pgn] [max_games]
"""

from __future__ import print_function
import chess
import chess.pgn
import chess.engine
from pathlib import Path
import sys

# Configuration
STOCKFISH = r"S:\Programming\Chess Engines\Tournament Engines\downloaded_engines\stockfish\stockfish-windows-x86-64-avx2.exe"
DEPTH = 20  # Analysis depth
TIME_PER_MOVE = 1.0  # seconds

# Mistake thresholds (centipawns)
BLUNDER = 300
MISTAKE = 150
INACCURACY = 50


class GameAnnotator:
    """Annotates chess games with Stockfish analysis"""
    
    def __init__(self):
        print("Starting Stockfish: " + STOCKFISH)
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    
    def analyze_position(self, board):
        """Analyze position and return (score_cp, best_move)"""
        info = self.engine.analyse(board, chess.engine.Limit(depth=DEPTH, time=TIME_PER_MOVE))
        score = info["score"].white().score(mate_score=10000)
        
        # Convert to current player's perspective
        if board.turn == chess.BLACK:
            score = -score
        
        best_move = info.get("pv", [None])[0]
        return score, best_move
    
    def annotate_game(self, game):
        """Annotate a game with engine analysis"""
        annotated = chess.pgn.Game()
        annotated.headers.update(game.headers)
        annotated.headers["Annotator"] = "Stockfish 17.1"
        
        stats = {
            "white": {"blunders": 0, "mistakes": 0, "inaccuracies": 0, "acpl": []},
            "black": {"blunders": 0, "mistakes": 0, "inaccuracies": 0, "acpl": []}
        }
        
        board = game.board()
        node = annotated
        move_count = 0
        
        for move in game.mainline_moves():
            move_count += 1
            
            # Analyze before move
            score_before, best_move = self.analyze_position(board)
            
            # Make move
            board.push(move)
            
            # Analyze after move
            score_after, _ = self.analyze_position(board)
            
            # Calculate loss from player's perspective
            # score_after is from opponent's perspective, so negate it
            loss = score_before - (-score_after)
            
            player = "white" if board.turn == chess.BLACK else "black"
            
            # Classify move quality
            if loss > BLUNDER:
                quality = "BLUNDER"
                stats[player]["blunders"] += 1
            elif loss > MISTAKE:
                quality = "MISTAKE"
                stats[player]["mistakes"] += 1
            elif loss > INACCURACY:
                quality = "INACCURACY"
                stats[player]["inaccuracies"] += 1
            else:
                quality = "good"
            
            # Track centipawn loss
            if loss > 0:
                stats[player]["acpl"].append(loss)
            
            # Add move with annotation
            node = node.add_variation(move)
            
            # Build comment
            comment = "[%+.2f]" % (score_before/100.0,)
            
            if quality != "good":
                comment += " " + quality + "!"
                if best_move and best_move != move:
                    comment += " Best: " + best_move.uci()
            
            if abs(loss) > 10:
                comment += " (D%+.2f)" % (-loss/100.0,)
            
            node.comment = comment
            
            # Progress indicator
            if move_count % 5 == 0:
                print("  Move %d..." % move_count, end='\r')
        
        print("  Analyzed %d moves" % move_count)
        
        # Calculate ACPL
        for color in ["white", "black"]:
            if stats[color]["acpl"]:
                avg = sum(stats[color]["acpl"]) / len(stats[color]["acpl"])
                stats[color]["acpl"] = round(avg, 1)
            else:
                stats[color]["acpl"] = 0
        
        return annotated, stats
    
    def close(self):
        """Close the engine"""
        if self.engine:
            self.engine.quit()


def main():
    """Main function"""
    print("=" * 80)
    print("PGN Game Annotator with Stockfish 17.1")
    print("=" * 80)
    
    # Parse arguments
    if len(sys.argv) > 1:
        input_pgn = sys.argv[1]
    else:
        input_pgn = r"s:\Programming\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot\lichess_v7p3r_bot_2025-12-21.pgn"
    
    if len(sys.argv) > 2:
        output_pgn = sys.argv[2]
    else:
        output_pgn = input_pgn.replace('.pgn', '_annotated.pgn')
        if output_pgn == input_pgn:
            output_pgn = input_pgn.replace('.pgn', '') + '_annotated.pgn'
    
    max_games = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    print("\nInput:  " + input_pgn)
    print("Output: " + output_pgn)
    print("Max games: " + str(max_games))
    print("Analysis depth: " + str(DEPTH))
    print()
    
    # Create annotator
    annotator = GameAnnotator()
    
    try:
        # Process games
        with open(input_pgn) as f_in, open(output_pgn, 'w') as f_out:
            game_num = 0
            
            while game_num < max_games:
                game = chess.pgn.read_game(f_in)
                if not game:
                    break
                
                game_num += 1
                white = game.headers.get('White', 'Unknown')
                black = game.headers.get('Black', 'Unknown')
                result = game.headers.get('Result', '*')
                
                print("\nGame %d: %s vs %s (%s)" % (game_num, white, black, result))
                
                # Annotate game
                annotated, stats = annotator.annotate_game(game)
                
                # Print statistics
                print("  White: %dB %dM %dI (ACPL: %.1f)" % (
                    stats['white']['blunders'], stats['white']['mistakes'], 
                    stats['white']['inaccuracies'], stats['white']['acpl']))
                print("  Black: %dB %dM %dI (ACPL: %.1f)" % (
                    stats['black']['blunders'], stats['black']['mistakes'], 
                    stats['black']['inaccuracies'], stats['black']['acpl']))
                
                # Write to output
                print(annotated, file=f_out, end="\n\n")
        
        print("\n" + "=" * 80)
        print("Analysis complete! Analyzed %d games" % game_num)
        print("Annotated PGN saved to: " + output_pgn)
        print("=" * 80)
    
    finally:
        annotator.close()


if __name__ == "__main__":
    main()
