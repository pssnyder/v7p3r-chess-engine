"""
Blunder Theme Analyzer
Analyzes annotated PGN games to identify tactical/positional themes in mistakes
and determines what V7P3R evaluation components should have caught them.

Extracts blunders/mistakes and classifies them by chess theme:
- King safety (exposed king, back rank, mating attacks)
- Tactical oversights (hanging pieces, forks, pins, skewers)
- Pawn structure (weak squares, isolated pawns, passed pawns)
- Piece coordination (trapped pieces, bad pieces)
- Material imbalance handling
- Endgame technique (king activity, opposition, zugzwang)
"""

from __future__ import print_function
import chess
import chess.pgn
import re
from collections import defaultdict

# Path to annotated PGN
ANNOTATED_PGN = r"s:\Programming\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot\lichess_v7p3r_bot_2025-12-21_annotated.pgn"

class BlunderThemeAnalyzer:
    def __init__(self):
        self.themes = defaultdict(list)
        self.v7p3r_mistakes = []
        
    def classify_position(self, board, move, eval_loss, best_move_san):
        """Classify the tactical/positional theme of a mistake"""
        themes = []
        
        # Determine material count
        piece_count = len(board.piece_map())
        is_endgame = piece_count <= 10
        is_middlegame = 10 < piece_count <= 20
        
        # Get the move that was played
        try:
            move_obj = board.parse_san(move)
        except:
            return ["parsing_error"]
        
        # Check king safety themes
        if self.is_king_safety_issue(board, move_obj, eval_loss):
            themes.append("king_safety")
            
        # Check for hanging pieces
        if self.is_hanging_piece(board, move_obj):
            themes.append("hanging_piece")
            
        # Check for tactical blunders (checks, captures, threats)
        if board.is_check():
            themes.append("check_related")
            
        # Check for pawn structure issues
        if move_obj.promotion or board.piece_at(move_obj.from_square).piece_type == chess.PAWN:
            themes.append("pawn_play")
            
        # Check for piece coordination
        if self.is_piece_coordination_issue(board, move_obj):
            themes.append("piece_coordination")
            
        # Endgame-specific themes
        if is_endgame:
            themes.append("endgame_technique")
            
        # Material imbalance
        if abs(self.material_balance(board)) > 300:
            themes.append("material_imbalance")
            
        # If evaluation drop is catastrophic (>5 pawns), likely tactical
        if eval_loss > 500:
            themes.append("tactical_blunder")
            
        return themes if themes else ["unknown"]
    
    def is_king_safety_issue(self, board, move, eval_loss):
        """Check if mistake relates to king safety"""
        # Castling rights lost
        if board.has_kingside_castling_rights(board.turn) or board.has_queenside_castling_rights(board.turn):
            if eval_loss > 100:
                return True
        
        # King exposed after move
        board_copy = board.copy()
        board_copy.push(move)
        king_square = board_copy.king(board.turn)
        
        # Count attackers around king
        attackers = 0
        for square in chess.SQUARES:
            if board_copy.is_attacked_by(not board.turn, square):
                if chess.square_distance(square, king_square) <= 2:
                    attackers += 1
        
        return attackers > 5
    
    def is_hanging_piece(self, board, move):
        """Check if move hangs a piece"""
        board_copy = board.copy()
        board_copy.push(move)
        
        # Check if moved piece is now attacked
        if board_copy.is_attacked_by(not board.turn, move.to_square):
            piece = board_copy.piece_at(move.to_square)
            if piece and piece.piece_type > chess.PAWN:
                # Check if adequately defended
                attackers = len(board_copy.attackers(not board.turn, move.to_square))
                defenders = len(board_copy.attackers(board.turn, move.to_square))
                if attackers > defenders:
                    return True
        return False
    
    def is_piece_coordination_issue(self, board, move):
        """Check if move creates poor piece coordination"""
        # Piece moved to edge of board
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.KNIGHT:
            # Knights on rim are dim
            if to_file in [0, 7] or to_rank in [0, 7]:
                return True
        
        return False
    
    def material_balance(self, board):
        """Calculate material balance in centipawns"""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900
        }
        
        balance = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                balance += value if piece.color == chess.WHITE else -value
        
        return balance
    
    def analyze_game(self, game):
        """Analyze a single game for V7P3R mistakes"""
        board = game.board()
        move_num = 0
        
        white = game.headers.get('White', 'Unknown')
        black = game.headers.get('Black', 'Unknown')
        v7p3r_color = None
        
        if 'v7p3r' in white.lower():
            v7p3r_color = chess.WHITE
        elif 'v7p3r' in black.lower():
            v7p3r_color = chess.BLACK
        else:
            return  # Not a V7P3R game
        
        for node in game.mainline():
            move_num += 1
            move = node.move
            comment = node.comment
            
            # Parse comment for mistake classification
            if not comment:
                board.push(move)
                continue
            
            # Check if this move is by V7P3R
            is_v7p3r_move = (board.turn == v7p3r_color)
            
            if is_v7p3r_move and ('BLUNDER' in comment or 'MISTAKE' in comment):
                # Extract evaluation loss
                delta_match = re.search(r'\(D[-+]?(\d+\.?\d*)\)', comment)
                eval_loss = float(delta_match.group(1)) if delta_match else 0
                
                # Extract best move
                best_match = re.search(r'Best: ([a-h1-8\-=+#NBRQKO]+)', comment)
                best_move = best_match.group(1) if best_match else "unknown"
                
                # Classify theme
                move_san = board.san(move)
                themes = self.classify_position(board, move_san, eval_loss, best_move)
                
                mistake_type = 'BLUNDER' if 'BLUNDER' in comment else 'MISTAKE'
                
                mistake_data = {
                    'game': game.headers.get('Site', 'Unknown'),
                    'move_num': move_num,
                    'color': 'White' if v7p3r_color == chess.WHITE else 'Black',
                    'move': move_san,
                    'best_move': best_move,
                    'eval_loss': eval_loss,
                    'type': mistake_type,
                    'themes': themes,
                    'fen': board.fen(),
                    'comment': comment
                }
                
                self.v7p3r_mistakes.append(mistake_data)
                
                for theme in themes:
                    self.themes[theme].append(mistake_data)
            
            board.push(move)
    
    def generate_report(self):
        """Generate thematic analysis report"""
        print("=" * 80)
        print("V7P3R BLUNDER THEME ANALYSIS")
        print("=" * 80)
        print("\nTotal V7P3R Mistakes: %d" % len(self.v7p3r_mistakes))
        print("  Blunders: %d" % len([m for m in self.v7p3r_mistakes if m['type'] == 'BLUNDER']))
        print("  Mistakes: %d" % len([m for m in self.v7p3r_mistakes if m['type'] == 'MISTAKE']))
        print()
        
        # Theme breakdown
        print("=" * 80)
        print("THEME BREAKDOWN")
        print("=" * 80)
        sorted_themes = sorted(self.themes.items(), key=lambda x: len(x[1]), reverse=True)
        
        for theme, mistakes in sorted_themes:
            print("\n%s: %d occurrences" % (theme.upper().replace('_', ' '), len(mistakes)))
            print("-" * 80)
            
            # Show top 3 worst mistakes in this theme
            worst_mistakes = sorted(mistakes, key=lambda x: x['eval_loss'], reverse=True)[:3]
            for i, m in enumerate(worst_mistakes, 1):
                print("  %d. Move %d (%s): %s -> %s (Loss: %.0fcp)" % (
                    i, m['move_num'], m['color'], m['move'], m['best_move'], m['eval_loss']))
                print("     Game: %s" % m['game'])
                print("     FEN: %s" % m['fen'])
            print()
        
        # V7P3R evaluation gap analysis
        print("=" * 80)
        print("V7P3R EVALUATION GAP ANALYSIS")
        print("=" * 80)
        
        self.analyze_evaluation_gaps(sorted_themes)
    
    def analyze_evaluation_gaps(self, sorted_themes):
        """Identify what V7P3R is missing in its evaluation"""
        
        recommendations = {
            'king_safety': [
                "King safety evaluation appears weak",
                "Check pawn shield evaluation (bonus for pawns in front of king)",
                "Verify king zone attack detection (squares around king under attack)",
                "Review castling rights bonus - may be too low",
                "Add penalty for exposed king in middlegame"
            ],
            'tactical_blunder': [
                "Tactical vision depth insufficient",
                "Quiescence search may not be deep enough",
                "SEE (Static Exchange Evaluation) may be missing or broken",
                "Check threat detection before making moves",
                "Review move ordering - tactics should be searched first"
            ],
            'hanging_piece': [
                "Basic piece safety not detected",
                "Attacked piece evaluation missing or too weak",
                "Need to check: does PST override safety?",
                "Verify defended/attacked piece counting",
                "May need dedicated 'hanging piece' check"
            ],
            'endgame_technique': [
                "Endgame evaluation needs tuning",
                "King activity bonus may be too low",
                "Passed pawn evaluation weak",
                "Opposition and zugzwang not recognized",
                "Review endgame thresholds and bonuses"
            ],
            'pawn_play': [
                "Pawn structure evaluation incomplete",
                "Passed pawn bonus may be too low",
                "Isolated pawn penalty missing or weak",
                "Doubled pawn detection issues",
                "Pawn chains not properly valued"
            ],
            'piece_coordination': [
                "Piece mobility evaluation weak",
                "Bad piece penalties missing (knights on rim, blocked bishops)",
                "Piece harmony not considered",
                "Trapped piece detection needed",
                "Review piece-square tables for edge penalties"
            ],
            'material_imbalance': [
                "Material counting is basic",
                "Exchange evaluation (R vs B+N) may be off",
                "Queen vs pieces imbalance not handled well",
                "Need compensation evaluation for sacrifices",
                "Review piece values in different game phases"
            ]
        }
        
        for theme, mistakes in sorted_themes:
            if theme in recommendations and len(mistakes) > 2:
                print("\n" + theme.upper().replace('_', ' ') + ":")
                for rec in recommendations[theme]:
                    print("  - " + rec)
        
        print("\n" + "=" * 80)
        print("PRIORITY FIXES (by frequency):")
        print("=" * 80)
        for i, (theme, mistakes) in enumerate(sorted_themes[:5], 1):
            print("%d. %s (%d mistakes)" % (i, theme.upper().replace('_', ' '), len(mistakes)))


def main():
    analyzer = BlunderThemeAnalyzer()
    
    print("Reading annotated PGN...")
    with open(ANNOTATED_PGN) as pgn_file:
        game_count = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            game_count += 1
            print("  Analyzing game %d..." % game_count)
            analyzer.analyze_game(game)
    
    print("\nGenerating theme analysis report...\n")
    analyzer.generate_report()


if __name__ == '__main__':
    main()
