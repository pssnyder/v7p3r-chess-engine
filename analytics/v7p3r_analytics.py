"""
V7P3R Chess Analytics System
Automated game analysis using Stockfish with theme detection and performance insights.
"""
import os
import chess
import chess.pgn
import chess.engine
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MoveAnalysis:
    """Analysis of a single move."""
    move_number: int
    move_san: str
    move_uci: str
    color: str
    evaluation: float  # Centipawns
    best_move: str
    best_eval: float
    eval_diff: float  # How much worse than best move
    classification: str  # "best", "excellent", "good", "inaccuracy", "mistake", "blunder", "??blunder"
    themes: List[str] = field(default_factory=list)
    pv_line: List[str] = field(default_factory=list)
    depth: int = 20


@dataclass
class ThemeDetection:
    """Chess themes detected in the game."""
    castling_king_side: int = 0
    castling_queen_side: int = 0
    fianchetto: int = 0
    isolated_pawns: int = 0
    doubled_pawns: int = 0
    passed_pawns: int = 0
    bishop_pair: bool = False
    knight_outpost: int = 0
    rook_open_file: int = 0
    rook_seventh_rank: int = 0
    battery: int = 0  # Queen/Rook on same file/rank
    pin: int = 0
    skewer: int = 0
    fork: int = 0
    discovered_attack: int = 0
    mate_threats: List[int] = field(default_factory=list)  # Move numbers with mate threats
    
    def to_dict(self) -> Dict:
        return {
            "castling": {"kingside": self.castling_king_side, "queenside": self.castling_queen_side},
            "pawn_structure": {
                "isolated": self.isolated_pawns,
                "doubled": self.doubled_pawns,
                "passed": self.passed_pawns
            },
            "pieces": {
                "bishop_pair": self.bishop_pair,
                "knight_outpost": self.knight_outpost,
                "rook_open_file": self.rook_open_file,
                "rook_seventh": self.rook_seventh_rank
            },
            "tactics": {
                "battery": self.battery,
                "pin": self.pin,
                "skewer": self.skewer,
                "fork": self.fork,
                "discovered_attack": self.discovered_attack,
                "mate_threats": len(self.mate_threats)
            }
        }


@dataclass
class GameAnalysisReport:
    """Complete analysis report for a single game."""
    game_id: str
    date: str
    result: str
    opponent: str
    color: str  # "white" or "black"
    time_control: str
    opening: str
    moves: List[MoveAnalysis] = field(default_factory=list)
    themes: ThemeDetection = field(default_factory=ThemeDetection)
    
    # Performance metrics
    average_centipawn_loss: float = 0.0
    best_moves: int = 0
    excellent_moves: int = 0
    good_moves: int = 0
    inaccuracies: int = 0
    mistakes: int = 0
    blunders: int = 0
    critical_blunders: int = 0
    
    # Best move alignment (top 5 moves)
    top1_alignment: float = 0.0  # % of times our move was top 1
    top3_alignment: float = 0.0
    top5_alignment: float = 0.0
    
    def calculate_metrics(self):
        """Calculate all performance metrics from move analysis."""
        if not self.moves:
            return
        
        total_moves = len(self.moves)
        centipawn_losses = []
        top1_count = 0
        top3_count = 0
        top5_count = 0
        
        for move in self.moves:
            # Centipawn loss
            centipawn_losses.append(abs(move.eval_diff))
            
            # Classification counts
            if move.classification == "best":
                self.best_moves += 1
                top1_count += 1
                top3_count += 1
                top5_count += 1
            elif move.classification == "excellent":
                self.excellent_moves += 1
                top3_count += 1
                top5_count += 1
            elif move.classification == "good":
                self.good_moves += 1
                top5_count += 1
            elif move.classification == "inaccuracy":
                self.inaccuracies += 1
            elif move.classification == "mistake":
                self.mistakes += 1
            elif move.classification == "blunder":
                self.blunders += 1
            elif move.classification == "??blunder":
                self.critical_blunders += 1
        
        self.average_centipawn_loss = sum(centipawn_losses) / len(centipawn_losses) if centipawn_losses else 0
        self.top1_alignment = (top1_count / total_moves * 100) if total_moves > 0 else 0
        self.top3_alignment = (top3_count / total_moves * 100) if total_moves > 0 else 0
        self.top5_alignment = (top5_count / total_moves * 100) if total_moves > 0 else 0


class V7P3RAnalytics:
    """Main analytics engine for v7p3r games."""
    
    def __init__(self, stockfish_path: str, bot_username: str = "v7p3r_bot"):
        """
        Initialize analytics system.
        
        Args:
            stockfish_path: Path to Stockfish executable
            bot_username: Bot's username on Lichess
        """
        self.stockfish_path = stockfish_path
        self.bot_username = bot_username
        self.engine: Optional[chess.engine.SimpleEngine] = None
        self.analysis_depth = 20
        self.analysis_time = 0.5  # seconds per move
        
    def __enter__(self):
        """Context manager entry - start engine."""
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        logger.info(f"Stockfish engine started: {self.engine.id}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close engine."""
        if self.engine:
            self.engine.quit()
            logger.info("Stockfish engine closed")
    
    def analyze_game(self, pgn_path: str) -> Optional[GameAnalysisReport]:
        """
        Analyze a single PGN game file.
        
        Args:
            pgn_path: Path to PGN file
            
        Returns:
            GameAnalysisReport or None if analysis fails
        """
        if not self.engine:
            raise RuntimeError("Engine not started. Use context manager: 'with V7P3RAnalytics(...) as analytics:'")
        
        try:
            with open(pgn_path) as pgn_file:
                game = chess.pgn.read_game(pgn_file)
            
            if not game:
                logger.warning(f"Could not parse game from {pgn_path}")
                return None
            
            # Extract game metadata
            headers = game.headers
            white = headers.get("White", "Unknown")
            black = headers.get("Black", "Unknown")
            
            # Determine our color
            if white.lower() == self.bot_username.lower():
                our_color = "white"
                opponent = black
            elif black.lower() == self.bot_username.lower():
                our_color = "black"
                opponent = white
            else:
                logger.warning(f"Bot {self.bot_username} not found in game {pgn_path}")
                return None
            
            report = GameAnalysisReport(
                game_id=headers.get("GameId", headers.get("Site", "unknown").split("/")[-1]),
                date=headers.get("UTCDate", "unknown"),
                result=headers.get("Result", "*"),
                opponent=opponent,
                color=our_color,
                time_control=headers.get("TimeControl", "unknown"),
                opening=headers.get("Opening", "unknown")
            )
            
            # Analyze each move
            board = game.board()
            move_number = 0
            
            for node in game.mainline():
                move = node.move
                move_number += 1
                
                # Only analyze our moves
                if (our_color == "white" and board.turn == chess.WHITE) or \
                   (our_color == "black" and board.turn == chess.BLACK):
                    
                    move_analysis = self._analyze_move(board, move, move_number)
                    if move_analysis:
                        report.moves.append(move_analysis)
                
                # Make the move
                board.push(move)
            
            # Detect themes
            report.themes = self._detect_themes(game)
            
            # Calculate metrics
            report.calculate_metrics()
            
            logger.info(f"Analyzed game {report.game_id}: {len(report.moves)} moves")
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing game {pgn_path}: {e}")
            return None
    
    def _analyze_move(self, board: chess.Board, move: chess.Move, move_number: int) -> Optional[MoveAnalysis]:
        """Analyze a single move with Stockfish."""
        try:
            # Get evaluation before the move
            info_before = self.engine.analyse(board, chess.engine.Limit(depth=self.analysis_depth))
            
            # Get best move and its evaluation
            best_move_info = self.engine.analyse(
                board, 
                chess.engine.Limit(depth=self.analysis_depth),
                multipv=5  # Get top 5 moves
            )
            
            # Make the move temporarily
            board_after = board.copy()
            board_after.push(move)
            info_after = self.engine.analyse(board_after, chess.engine.Limit(depth=self.analysis_depth))
            
            # Extract evaluations
            score_after = info_after.get("score", chess.engine.Cp(0)).relative
            best_score = best_move_info[0].get("score", chess.engine.Cp(0)).relative
            best_move = best_move_info[0].get("pv", [chess.Move.null()])[0]
            
            # Convert to centipawns
            eval_after_cp = self._score_to_cp(score_after)
            best_eval_cp = self._score_to_cp(best_score)
            eval_diff = best_eval_cp - eval_after_cp
            
            # Classify move
            classification = self._classify_move(eval_diff, score_after)
            
            # Extract PV
            pv_line = [board.san(m) for m in best_move_info[0].get("pv", [])[:5]]
            
            return MoveAnalysis(
                move_number=move_number,
                move_san=board.san(move),
                move_uci=move.uci(),
                color="white" if board.turn == chess.WHITE else "black",
                evaluation=eval_after_cp,
                best_move=board.san(best_move) if best_move != move else board.san(move),
                best_eval=best_eval_cp,
                eval_diff=eval_diff,
                classification=classification,
                pv_line=pv_line,
                depth=self.analysis_depth
            )
            
        except Exception as e:
            logger.debug(f"Error analyzing move {move_number}: {e}")
            return None
    
    def _score_to_cp(self, score: chess.engine.Score) -> float:
        """Convert chess.engine.Score to centipawns."""
        if score.is_mate():
            # Mate score: large value with sign
            mate_in = score.mate()
            return 10000 * (1 if mate_in > 0 else -1) - mate_in
        else:
            return float(score.score())
    
    def _classify_move(self, eval_diff: float, score_after: chess.engine.Score) -> str:
        """Classify move quality based on evaluation difference."""
        # If we found mate, it's best
        if score_after.is_mate() and score_after.mate() > 0:
            return "best"
        
        # Otherwise classify by centipawn loss
        if eval_diff <= 10:
            return "best"
        elif eval_diff <= 25:
            return "excellent"
        elif eval_diff <= 50:
            return "good"
        elif eval_diff <= 100:
            return "inaccuracy"
        elif eval_diff <= 200:
            return "mistake"
        elif eval_diff <= 400:
            return "blunder"
        else:
            return "??blunder"
    
    def _detect_themes(self, game: chess.pgn.Game) -> ThemeDetection:
        """Detect chess themes throughout the game."""
        themes = ThemeDetection()
        board = game.board()
        
        for node in game.mainline():
            move = node.move
            
            # Castling
            if board.is_castling(move):
                if move.to_square > move.from_square:  # King side
                    themes.castling_king_side += 1
                else:
                    themes.castling_queen_side += 1
            
            # Make move for position analysis
            board.push(move)
            
            # Analyze position themes (simplified detection)
            themes.isolated_pawns += self._count_isolated_pawns(board)
            themes.passed_pawns += self._count_passed_pawns(board)
            
            # Bishop pair
            if len(board.pieces(chess.BISHOP, chess.WHITE)) == 2:
                themes.bishop_pair = True
            if len(board.pieces(chess.BISHOP, chess.BLACK)) == 2:
                themes.bishop_pair = True
        
        return themes
    
    def _count_isolated_pawns(self, board: chess.Board) -> int:
        """Count isolated pawns on the board."""
        count = 0
        for color in [chess.WHITE, chess.BLACK]:
            pawns_bb = board.pieces(chess.PAWN, color)
            for pawn_square in list(pawns_bb):
                file = chess.square_file(pawn_square)
                # Check adjacent files for pawns of same color
                has_neighbor = False
                for adj_file in [file - 1, file + 1]:
                    if 0 <= adj_file <= 7:
                        file_mask = chess.BB_FILES[adj_file]
                        if pawns_bb & file_mask:
                            has_neighbor = True
                            break
                if not has_neighbor:
                    count += 1
        return count
    
    def _count_passed_pawns(self, board: chess.Board) -> int:
        """Count passed pawns on the board."""
        count = 0
        for color in [chess.WHITE, chess.BLACK]:
            pawns_bb = board.pieces(chess.PAWN, color)
            enemy_pawns_bb = board.pieces(chess.PAWN, not color)
            
            for pawn_square in list(pawns_bb):
                file = chess.square_file(pawn_square)
                rank = chess.square_rank(pawn_square)
                
                # Check if pawn is passed
                is_passed = True
                for adj_file in [file - 1, file, file + 1]:
                    if 0 <= adj_file <= 7:
                        # Check squares ahead
                        if color == chess.WHITE:
                            # For white, check ranks above current rank
                            ahead_squares = chess.BB_FILES[adj_file] & chess.BB_RANK_8
                            for r in range(rank + 1, 8):
                                ahead_squares |= chess.BB_FILES[adj_file] & chess.BB_RANKS[r]
                        else:
                            # For black, check ranks below current rank
                            ahead_squares = chess.BB_FILES[adj_file] & chess.BB_RANK_1
                            for r in range(0, rank):
                                ahead_squares |= chess.BB_FILES[adj_file] & chess.BB_RANKS[r]
                        
                        if enemy_pawns_bb & ahead_squares:
                            is_passed = False
                            break
                
                if is_passed:
                    count += 1
        
        return count


if __name__ == "__main__":
    # Test the analyzer
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python v7p3r_analytics.py <stockfish_path> <pgn_file>")
        sys.exit(1)
    
    stockfish_path = sys.argv[1]
    pgn_file = sys.argv[2]
    
    with V7P3RAnalytics(stockfish_path) as analytics:
        report = analytics.analyze_game(pgn_file)
        
        if report:
            print(f"\nGame Analysis: {report.game_id}")
            print(f"Opponent: {report.opponent} | Result: {report.result}")
            print(f"Opening: {report.opening}")
            print(f"\nPerformance:")
            print(f"  Avg CPL: {report.average_centipawn_loss:.1f}")
            print(f"  Best moves: {report.best_moves}")
            print(f"  Excellent: {report.excellent_moves}")
            print(f"  Good: {report.good_moves}")
            print(f"  Inaccuracies: {report.inaccuracies}")
            print(f"  Mistakes: {report.mistakes}")
            print(f"  Blunders: {report.blunders}")
            print(f"  Critical: {report.critical_blunders}")
            print(f"\nAlignment with Stockfish:")
            print(f"  Top 1: {report.top1_alignment:.1f}%")
            print(f"  Top 3: {report.top3_alignment:.1f}%")
            print(f"  Top 5: {report.top5_alignment:.1f}%")
            print(f"\nThemes detected:")
            print(json.dumps(report.themes.to_dict(), indent=2))
