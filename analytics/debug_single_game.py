#!/usr/bin/env python3
"""
Debug single game analysis to verify theme detection
"""

import chess
import chess.engine
import chess.pgn
from io import StringIO

# Simple PGN from weekend analysis (M2xRGcyK - v7p3r_bot win)
PGN = """[Event "casual rapid game"]
[Site "https://lichess.org/M2xRGcyK"]
[Date "2025.12.14"]
[White "v7p3r_bot"]
[Black "NyaaBot"]
[Result "1-0"]

1. d4 h6 2. Nc3 e5 3. dxe5 c6 4. Nf3 Nf6 5. exf6 gxf6 6. Qd4 Bg7 
7. Qe4+ Kf8 8. Qd4 Rh7 9. e3 Qe8 10. Ne4 b5 11. Nd6 b4 12. Nxe8 Kxe8 
13. Qe4+ Kf8 14. Qxh7 Ke8 15. Qxg7 b3 16. axb3 f5 17. Qh8+ Ke7 
18. Qxc8 Kf6 19. Qb7 Ke6 20. Qxa8 Kd6 21. Qxb8+ Ke7 22. Rxa7 Kf6 
23. Rxd7 Ke6 24. Qd6# 1-0"""

def analyze_position_manually(board: chess.Board) -> dict:
    """Manually detect themes in position."""
    themes = {
        'isolated_pawns': 0,
        'doubled_pawns': 0,
        'passed_pawns': 0,
        'pins': 0,
        'forks': 0,
        'open_files_with_rooks': 0,
        'seventh_rank_rooks': 0
    }
    
    # Count isolated pawns (white only since we're v7p3r_bot)
    for file in range(8):
        white_pawn_on_file = False
        for rank in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                white_pawn_on_file = True
                
                # Check if isolated (no friendly pawns on adjacent files)
                has_neighbor = False
                for adj_file in [file - 1, file + 1]:
                    if 0 <= adj_file < 8:
                        for adj_rank in range(8):
                            adj_square = chess.square(adj_file, adj_rank)
                            adj_piece = board.piece_at(adj_square)
                            if adj_piece and adj_piece.piece_type == chess.PAWN and adj_piece.color == chess.WHITE:
                                has_neighbor = True
                                break
                
                if not has_neighbor:
                    themes['isolated_pawns'] += 1
    
    # Count passed pawns (simplified - check if no enemy pawns ahead or on adjacent files)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            is_passed = True
            for adj_file in [file - 1, file, file + 1]:
                if 0 <= adj_file < 8:
                    for check_rank in range(rank + 1, 8):  # Check ahead
                        check_square = chess.square(adj_file, check_rank)
                        check_piece = board.piece_at(check_square)
                        if check_piece and check_piece.piece_type == chess.PAWN and check_piece.color == chess.BLACK:
                            is_passed = False
                            break
            
            if is_passed and rank > 1:  # Don't count 2nd rank as passed
                themes['passed_pawns'] += 1
    
    # Count rooks on 7th rank
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.ROOK and piece.color == chess.WHITE:
            rank = chess.square_rank(square)
            if rank == 6:  # 7th rank (0-indexed)
                themes['seventh_rank_rooks'] += 1
    
    return themes

def main():
    print("=" * 80)
    print("DEBUG: Single Game Analysis - Manual Theme Detection")
    print("=" * 80)
    print()
    
    # Parse PGN
    pgn = StringIO(PGN)
    game = chess.pgn.read_game(pgn)
    
    print(f"Game: {game.headers.get('Event')}")
    print(f"White: {game.headers.get('White')} (v7p3r_bot)")
    print(f"Black: {game.headers.get('Black')}")
    print(f"Result: {game.headers.get('Result')}")
    print(f"Moves: {len(list(game.mainline_moves()))} ply")
    print()
    
    # Walk through game and count themes
    board = game.board()
    total_themes = {
        'isolated_pawns': 0,
        'doubled_pawns': 0,
        'passed_pawns': 0,
        'pins': 0,
        'forks': 0,
        'seventh_rank_rooks': 0
    }
    
    move_count = 0
    for move in game.mainline_moves():
        board.push(move)
        move_count += 1
        
        # Only analyze after v7p3r_bot's moves (White)
        if board.turn == chess.BLACK:  # Just after White's move
            themes = analyze_position_manually(board)
            for key in total_themes:
                total_themes[key] += themes[key]
    
    print("THEME COUNTS (cumulative across all v7p3r_bot positions):")
    print("-" * 80)
    for theme, count in total_themes.items():
        per_move = count / (move_count // 2) if move_count > 0 else 0
        print(f"  {theme}: {count} total ({per_move:.2f} avg per position)")
    print()
    
    # Compare to analytics result
    print("ANALYTICS REPORTED FOR THIS GAME:")
    print("-" * 80)
    print("  isolated_pawns: 129")
    print("  passed_pawns: 0")
    print("  seventh_rank_rooks: 0")
    print()
    
    # Final position analysis
    print("FINAL POSITION ANALYSIS:")
    print("-" * 80)
    print(board)
    print()
    final_themes = analyze_position_manually(board)
    print("Final position themes:")
    for theme, count in final_themes.items():
        print(f"  {theme}: {count}")
    print()
    
    print("=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    if total_themes['isolated_pawns'] < 20:
        print("✅ Manual count shows reasonable isolated pawn count")
        print("🔴 Analytics is BROKEN - reported 129 isolated pawns for this game!")
    else:
        print("⚠️ High isolated pawn count confirmed")
    print()

if __name__ == "__main__":
    main()
