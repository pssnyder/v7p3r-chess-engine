#!/usr/bin/env python3
"""
V7P3R Game Replay Analyzer
Based on Universal Puzzle Analyzer but adapted for post-game analysis

Analyzes historical V7P3R games to identify performance blind spots:
1. Processes PGN files from game directories
2. Extracts all V7P3R moves and analyzes them against Stockfish
3. Identifies positions where V7P3R chose moves outside Stockfish's top 5
4. Creates "weakness puzzles" from these poor positions
5. Analyzes V7P3R's current move ordering on these historical mistakes
6. Provides comprehensive blind spot analysis and improvement recommendations

This creates a feedback loop: Real Game Performance → Weakness Detection → Targeted Analysis → Engine Improvement
"""

import chess
import chess.pgn
import chess.engine
import os
import sys
import json
import time
import subprocess
import threading
import signal
import gc
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union, Iterator
from dataclasses import dataclass, field

# Add src to path for V7P3R analysis
current_dir = os.path.dirname(__file__)
src_path = os.path.join(current_dir, 'src')
if os.path.exists(src_path):
    sys.path.append(src_path)

try:
    from analyze_move_ordering import V7P3RMoveAnalyzer
    MOVE_ANALYZER_AVAILABLE = True
except ImportError:
    print("Warning: V7P3RMoveAnalyzer not available, move ordering analysis will be skipped")
    MOVE_ANALYZER_AVAILABLE = False


@dataclass
class GameMove:
    """Represents a move made in a game with context"""
    game_id: str
    move_number: int
    player: str
    move_san: str
    move_uci: str
    position_before_fen: str
    position_after_fen: str
    time_left: Optional[str] = None
    evaluation: Optional[int] = None


@dataclass
class WeaknessPosition:
    """Represents a position where V7P3R made a poor move"""
    game_id: str
    move_number: int
    fen_before_move: str
    opponent_last_move: str
    v7p3r_move: str
    v7p3r_move_rank: int  # 0 if not in top 5, 1-5 if in top 5
    stockfish_top_moves: List[Tuple[str, int]]  # (move, centipawn)
    stockfish_best_move: str
    centipawn_loss: int  # How much worse V7P3R's move was
    position_themes: List[str]  # Tactical themes if any
    time_pressure: bool = False
    material_balance: int = 0
    game_phase: str = "middlegame"  # opening, middlegame, endgame


class V7P3RGameReplayAnalyzer:
    """Analyzes V7P3R's historical game performance to identify weaknesses"""
    
    def __init__(self, 
                 stockfish_path: str = r"S:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester\engines\Stockfish\stockfish-windows-x86-64-avx2.exe",
                 v7p3r_engine_path: str = r"S:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\src\v7p3r_uci.py"):
        
        self.stockfish_path = stockfish_path
        self.v7p3r_engine_path = v7p3r_engine_path
        self.move_analyzer = V7P3RMoveAnalyzer() if MOVE_ANALYZER_AVAILABLE else None
        
        # Analysis results
        self.weakness_positions: List[WeaknessPosition] = []
        self.game_statistics = defaultdict(int)
        self.analysis_start_time = None
        
        # Verify paths
        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Stockfish not found: {stockfish_path}")
        if not os.path.exists(v7p3r_engine_path):
            print(f"Warning: V7P3R engine not found at {v7p3r_engine_path}, move ordering analysis will be skipped")
        
        print("🔍 V7P3R Game Replay Analyzer initialized")
        print(f"Stockfish: {stockfish_path}")
        print(f"V7P3R Engine: {v7p3r_engine_path}")
    
    def find_pgn_files(self, directory: str) -> List[str]:
        """Find all PGN files in the given directory"""
        pgn_files = []
        
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            return pgn_files
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pgn'):
                    pgn_files.append(os.path.join(root, file))
        
        return pgn_files
    
    def extract_moves_from_game(self, game: chess.pgn.Game, player_name: str) -> List[GameMove]:
        """Extract all moves made by the specified player from a game"""
        moves = []
        board = chess.Board()
        
        # Get game metadata
        game_id = f"{game.headers.get('Site', 'unknown')}_{game.headers.get('Date', 'unknown')}"
        white_player = game.headers.get('White', '').lower()
        black_player = game.headers.get('Black', '').lower()
        
        # Determine if player is white or black
        is_white = player_name.lower() in white_player
        is_black = player_name.lower() in black_player
        
        if not (is_white or is_black):
            return moves  # Player not in this game
        
        move_number = 1
        for node in game.mainline():
            position_before = board.fen()
            move = node.move
            
            # Check if this is our player's move
            is_our_turn = (board.turn == chess.WHITE and is_white) or (board.turn == chess.BLACK and is_black)
            
            if is_our_turn:
                # Extract timing information if available
                time_left = None
                if node.comment:
                    # Try to extract time from comment (format varies)
                    if 'clk' in node.comment:
                        import re
                        time_match = re.search(r'clk (\d+:\d+:\d+)', node.comment)
                        if time_match:
                            time_left = time_match.group(1)
                
                # Create GameMove object
                game_move = GameMove(
                    game_id=game_id,
                    move_number=move_number if board.turn == chess.WHITE else move_number,
                    player=player_name,
                    move_san=board.san(move),
                    move_uci=move.uci(),
                    position_before_fen=position_before,
                    position_after_fen="",  # Will be set after move
                    time_left=time_left
                )
                
                moves.append(game_move)
            
            # Make the move
            board.push(move)
            
            # Update position after move for our moves
            if is_our_turn and moves:
                moves[-1].position_after_fen = board.fen()
            
            # Update move number after black's move
            if board.turn == chess.WHITE:
                move_number += 1
        
        return moves
    
    def get_stockfish_analysis(self, fen: str, time_seconds: float = 3.0) -> List[Tuple[str, int]]:
        """Get Stockfish's top 5 moves with evaluations for a position"""
        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                board = chess.Board(fen)
                
                # Analyze with multiple PV lines
                result = engine.analyse(
                    board,
                    chess.engine.Limit(time=time_seconds),
                    multipv=5
                )
                
                moves_with_scores = []
                for analysis in result:
                    if 'pv' in analysis and analysis['pv']:
                        move = analysis['pv'][0].uci()
                # Convert score to centipawns
                score = analysis.get('score', chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))
                if score.is_mate():
                    # Mate scores
                    mate_value = score.white().mate()
                    if mate_value is not None:
                        cp_value = 10000 if mate_value > 0 else -10000
                    else:
                        cp_value = 0
                else:
                    cp_value = score.white().score() or 0
                
                moves_with_scores.append((move, cp_value))
                
                return moves_with_scores
                
        except Exception as e:
            print(f"Error getting Stockfish analysis for {fen}: {e}")
            return []
    
    def classify_position_themes(self, board: chess.Board) -> List[str]:
        """Classify the tactical/strategic themes present in a position"""
        themes = []
        
        # Basic position classification
        piece_count = len(board.piece_map())
        if piece_count <= 10:
            themes.append("endgame")
        elif piece_count >= 28:
            themes.append("opening")
        else:
            themes.append("middlegame")
        
        # Check for tactical elements
        if board.is_check():
            themes.append("check")
        
        # Look for hanging pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                attackers = board.attackers(not piece.color, square)
                defenders = board.attackers(piece.color, square)
                if len(attackers) > len(defenders):
                    themes.append("hanging_piece")
                    break
        
        # Check for captures available
        has_captures = any(board.is_capture(move) for move in board.legal_moves)
        if has_captures:
            themes.append("tactics_available")
        
        # Material imbalance
        white_material = sum(self._get_piece_value(piece.piece_type) 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.WHITE)
        black_material = sum(self._get_piece_value(piece.piece_type) 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.BLACK)
        
        imbalance = abs(white_material - black_material)
        if imbalance > 300:
            themes.append("material_imbalance")
        
        return themes
    
    def _get_piece_value(self, piece_type: int) -> int:
        """Get standard piece values"""
        values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        return values.get(piece_type, 0)
    
    def analyze_move_quality(self, game_move: GameMove) -> Optional[WeaknessPosition]:
        """Analyze the quality of a single move against Stockfish"""
        board = chess.Board(game_move.position_before_fen)
        
        # Get Stockfish's top moves
        stockfish_moves = self.get_stockfish_analysis(game_move.position_before_fen, 3.0)
        if not stockfish_moves:
            return None
        
        # Find V7P3R's move rank
        v7p3r_rank = 0
        v7p3r_score = None
        
        for rank, (sf_move, sf_score) in enumerate(stockfish_moves, 1):
            if sf_move == game_move.move_uci:
                v7p3r_rank = rank
                v7p3r_score = sf_score
                break
        
        # If V7P3R's move is not in top 5, it's a weakness
        if v7p3r_rank == 0 or v7p3r_rank > 3:  # Focus on moves not in top 3
            best_move, best_score = stockfish_moves[0]
            
            # Calculate centipawn loss
            if v7p3r_score is not None:
                centipawn_loss = abs(best_score - v7p3r_score)
            else:
                # Estimate based on rank or assume significant loss
                centipawn_loss = 200  # Default for moves not in top 5
            
            # Get opponent's last move
            board_copy = chess.Board(game_move.position_before_fen)
            if len(board_copy.move_stack) > 0:
                opponent_last_move = board_copy.peek().uci()
            else:
                opponent_last_move = "game_start"
            
            # Classify position themes
            themes = self.classify_position_themes(board)
            
            # Create weakness position
            weakness = WeaknessPosition(
                game_id=game_move.game_id,
                move_number=game_move.move_number,
                fen_before_move=game_move.position_before_fen,
                opponent_last_move=opponent_last_move,
                v7p3r_move=game_move.move_uci,
                v7p3r_move_rank=v7p3r_rank,
                stockfish_top_moves=stockfish_moves,
                stockfish_best_move=best_move,
                centipawn_loss=centipawn_loss,
                position_themes=themes,
                time_pressure=self._is_time_pressure(game_move.time_left),
                material_balance=self._calculate_material_balance(board),
                game_phase=self._determine_game_phase(board)
            )
            
            return weakness
        
        return None
    
    def _is_time_pressure(self, time_left: Optional[str]) -> bool:
        """Determine if player was in time pressure"""
        if not time_left:
            return False
        
        try:
            # Parse time format (e.g., "0:05:30")
            parts = time_left.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                total_seconds = hours * 3600 + minutes * 60 + seconds
                return total_seconds < 300  # Less than 5 minutes
        except:
            pass
        
        return False
    
    def _calculate_material_balance(self, board: chess.Board) -> int:
        """Calculate material balance (positive = white advantage)"""
        white_material = sum(self._get_piece_value(piece.piece_type) 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.WHITE)
        black_material = sum(self._get_piece_value(piece.piece_type) 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.BLACK)
        
        return white_material - black_material
    
    def _determine_game_phase(self, board: chess.Board) -> str:
        """Determine game phase based on material"""
        piece_count = len(board.piece_map())
        
        if piece_count >= 28:
            return "opening"
        elif piece_count <= 10:
            return "endgame"
        else:
            return "middlegame"
    
    def analyze_current_move_ordering(self, weakness: WeaknessPosition) -> Dict:
        """Analyze how current V7P3R would handle this position"""
        if not self.move_analyzer:
            return {"error": "Move analyzer not available"}
        
        try:
            board = chess.Board(weakness.fen_before_move)
            
            # Get current engine's move ordering for this position
            analysis = self.move_analyzer.compare_move_selection(board, top_n=8)
            
            # Check where the historical bad move ranks now
            engine_ordered_moves = analysis.get('engine_order', [])
            historical_move_rank = 0
            
            for rank, move in enumerate(engine_ordered_moves, 1):
                if move.uci() == weakness.v7p3r_move:
                    historical_move_rank = rank
                    break
            
            # Categorize the historical move
            historical_move_categories = self.move_analyzer.categorize_move(board, chess.Move.from_uci(weakness.v7p3r_move))
            
            # Categorize Stockfish's best move
            stockfish_best_categories = self.move_analyzer.categorize_move(board, chess.Move.from_uci(weakness.stockfish_best_move))
            
            return {
                "current_engine_rank": historical_move_rank,
                "current_top_moves": [move.uci() for move in engine_ordered_moves[:5]],
                "historical_move_categories": historical_move_categories,
                "stockfish_best_categories": stockfish_best_categories,
                "move_ordering_analysis": analysis,
                "improvement": historical_move_rank > 0 and historical_move_rank <= 3  # Improved if now in top 3
            }
            
        except Exception as e:
            return {"error": f"Move ordering analysis failed: {e}"}
    
    def process_game_directory(self, directory: str, player_name: str, 
                              max_games: Optional[int] = None, 
                              analyze_move_ordering: bool = True) -> Dict:
        """Process all PGN files in directory to find V7P3R weaknesses"""
        
        print(f"🎮 Processing games in: {directory}")
        print(f"Target player: {player_name}")
        print("=" * 60)
        
        self.analysis_start_time = time.time()
        
        # Find all PGN files
        pgn_files = self.find_pgn_files(directory)
        if not pgn_files:
            print(f"No PGN files found in {directory}")
            return {}
        
        print(f"Found {len(pgn_files)} PGN files")
        
        # Process each PGN file
        games_processed = 0
        moves_analyzed = 0
        weaknesses_found = 0
        
        for pgn_file in pgn_files:
            print(f"\n📁 Processing: {os.path.basename(pgn_file)}")
            
            try:
                with open(pgn_file, 'r', encoding='utf-8') as f:
                    while True:
                        if max_games and games_processed >= max_games:
                            break
                            
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break
                        
                        games_processed += 1
                        print(f"  Game {games_processed}: {game.headers.get('White', '?')} vs {game.headers.get('Black', '?')}")
                        
                        # Extract moves for our player
                        player_moves = self.extract_moves_from_game(game, player_name)
                        print(f"    {player_name} made {len(player_moves)} moves")
                        
                        # Analyze each move
                        for move in player_moves:
                            moves_analyzed += 1
                            weakness = self.analyze_move_quality(move)
                            
                            if weakness:
                                weaknesses_found += 1
                                self.weakness_positions.append(weakness)
                                
                                print(f"    ❌ Weakness found: Move {weakness.move_number} ({weakness.v7p3r_move}) "
                                      f"not in top {3 if weakness.v7p3r_move_rank == 0 else weakness.v7p3r_move_rank-1}, "
                                      f"lost {weakness.centipawn_loss}cp")
                        
                        # Show progress
                        if games_processed % 10 == 0:
                            elapsed = time.time() - self.analysis_start_time
                            rate = games_processed / elapsed * 60  # games per minute
                            print(f"    Progress: {games_processed} games, {weaknesses_found} weaknesses, {rate:.1f} games/min")
                
            except Exception as e:
                print(f"Error processing {pgn_file}: {e}")
                continue
        
        # Analyze current move ordering for weaknesses
        if analyze_move_ordering and self.weakness_positions:
            print(f"\n🔍 Analyzing current move ordering for {len(self.weakness_positions)} weakness positions...")
            
            for i, weakness in enumerate(self.weakness_positions[:20]):  # Limit to first 20 for performance
                print(f"  Analyzing weakness {i+1}/20...")
                ordering_analysis = self.analyze_current_move_ordering(weakness)
                weakness.current_analysis = ordering_analysis
        
        # Compile statistics
        total_time = time.time() - self.analysis_start_time
        
        results = {
            "summary": {
                "games_processed": games_processed,
                "moves_analyzed": moves_analyzed,
                "weaknesses_found": weaknesses_found,
                "weakness_rate": (weaknesses_found / moves_analyzed * 100) if moves_analyzed > 0 else 0,
                "analysis_time_minutes": total_time / 60,
                "games_per_minute": games_processed / (total_time / 60) if total_time > 0 else 0
            },
            "weakness_positions": self.weakness_positions,
            "pgn_files_processed": pgn_files
        }
        
        return results
    
    def generate_weakness_report(self, results: Dict) -> Dict:
        """Generate comprehensive weakness analysis report"""
        if not self.weakness_positions:
            return {"error": "No weakness positions found"}
        
        weaknesses = self.weakness_positions
        
        # Categorize weaknesses by theme
        theme_analysis = defaultdict(list)
        for weakness in weaknesses:
            for theme in weakness.position_themes:
                theme_analysis[theme].append(weakness)
        
        # Rank themes by frequency and severity
        theme_stats = {}
        for theme, positions in theme_analysis.items():
            avg_cp_loss = sum(p.centipawn_loss for p in positions) / len(positions)
            worst_cp_loss = max(p.centipawn_loss for p in positions)
            
            theme_stats[theme] = {
                "count": len(positions),
                "avg_centipawn_loss": avg_cp_loss,
                "worst_centipawn_loss": worst_cp_loss,
                "severity_score": len(positions) * avg_cp_loss  # Frequency * severity
            }
        
        # Game phase analysis
        phase_stats = defaultdict(list)
        for weakness in weaknesses:
            phase_stats[weakness.game_phase].append(weakness.centipawn_loss)
        
        phase_analysis = {}
        for phase, losses in phase_stats.items():
            phase_analysis[phase] = {
                "count": len(losses),
                "avg_loss": sum(losses) / len(losses),
                "worst_loss": max(losses)
            }
        
        # Time pressure analysis
        time_pressure_weaknesses = [w for w in weaknesses if w.time_pressure]
        normal_time_weaknesses = [w for w in weaknesses if not w.time_pressure]
        
        # Move ordering improvement analysis
        if hasattr(weaknesses[0], 'current_analysis'):
            improved_positions = sum(1 for w in weaknesses 
                                   if hasattr(w, 'current_analysis') 
                                   and w.current_analysis.get('improvement', False))
            improvement_rate = (improved_positions / len(weaknesses)) * 100
        else:
            improvement_rate = 0
        
        # Worst positions (highest centipawn loss)
        worst_positions = sorted(weaknesses, key=lambda x: x.centipawn_loss, reverse=True)[:10]
        
        # Most frequent mistake patterns
        mistake_patterns = defaultdict(int)
        for weakness in weaknesses:
            pattern = f"{weakness.game_phase}_{'+'.join(weakness.position_themes[:2])}"
            mistake_patterns[pattern] += 1
        
        report = {
            "weakness_summary": {
                "total_weaknesses": len(weaknesses),
                "avg_centipawn_loss": sum(w.centipawn_loss for w in weaknesses) / len(weaknesses),
                "worst_centipawn_loss": max(w.centipawn_loss for w in weaknesses),
                "time_pressure_rate": len(time_pressure_weaknesses) / len(weaknesses) * 100,
                "current_improvement_rate": improvement_rate
            },
            "theme_analysis": dict(sorted(theme_stats.items(), 
                                        key=lambda x: x[1]['severity_score'], 
                                        reverse=True)),
            "phase_analysis": phase_analysis,
            "worst_positions": [
                {
                    "game_id": w.game_id,
                    "move_number": w.move_number,
                    "fen": w.fen_before_move,
                    "v7p3r_move": w.v7p3r_move,
                    "stockfish_best": w.stockfish_best_move,
                    "centipawn_loss": w.centipawn_loss,
                    "themes": w.position_themes
                } for w in worst_positions
            ],
            "mistake_patterns": dict(sorted(mistake_patterns.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True)[:10]),
            "recommendations": self._generate_recommendations(theme_stats, phase_analysis)
        }
        
        return report
    
    def _generate_recommendations(self, theme_stats: Dict, phase_analysis: Dict) -> List[str]:
        """Generate improvement recommendations based on analysis"""
        recommendations = []
        
        # Theme-based recommendations
        top_themes = sorted(theme_stats.items(), 
                           key=lambda x: x[1]['severity_score'], 
                           reverse=True)[:3]
        
        for theme, stats in top_themes:
            if theme == "tactics_available":
                recommendations.append(f"HIGH PRIORITY: Improve tactical pattern recognition - "
                                     f"{stats['count']} missed tactical opportunities (avg {stats['avg_centipawn_loss']:.0f}cp loss)")
            elif theme == "endgame":
                recommendations.append(f"MEDIUM PRIORITY: Strengthen endgame play - "
                                     f"{stats['count']} endgame mistakes (avg {stats['avg_centipawn_loss']:.0f}cp loss)")
            elif theme == "material_imbalance":
                recommendations.append(f"HIGH PRIORITY: Improve evaluation in imbalanced positions - "
                                     f"{stats['count']} errors (avg {stats['avg_centipawn_loss']:.0f}cp loss)")
        
        # Phase-based recommendations
        worst_phase = max(phase_analysis.items(), key=lambda x: x[1]['avg_loss'])
        recommendations.append(f"FOCUS AREA: {worst_phase[0].title()} phase needs attention - "
                             f"avg {worst_phase[1]['avg_loss']:.0f}cp loss per mistake")
        
        return recommendations
    
    def export_weakness_puzzles(self, output_file: str, max_puzzles: int = 50) -> str:
        """Export weakness positions as puzzle format for training"""
        if not self.weakness_positions:
            return "No weakness positions to export"
        
        # Sort by centipawn loss and take worst positions
        worst_positions = sorted(self.weakness_positions, 
                                key=lambda x: x.centipawn_loss, 
                                reverse=True)[:max_puzzles]
        
        puzzles = []
        for i, weakness in enumerate(worst_positions, 1):
            puzzle = {
                "id": f"v7p3r_weakness_{i}",
                "fen": weakness.fen_before_move,
                "moves": weakness.stockfish_best_move,  # Just the best move for now
                "rating": min(2000, 800 + weakness.centipawn_loss // 10),  # Estimate puzzle rating
                "themes": " ".join(weakness.position_themes),
                "game_context": {
                    "game_id": weakness.game_id,
                    "move_number": weakness.move_number,
                    "v7p3r_chose": weakness.v7p3r_move,
                    "centipawn_loss": weakness.centipawn_loss,
                    "opponent_last_move": weakness.opponent_last_move
                }
            }
            puzzles.append(puzzle)
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump({
                "weakness_puzzles": puzzles,
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "total_weaknesses": len(self.weakness_positions),
                    "exported_count": len(puzzles),
                    "source": "V7P3R Game Replay Analysis"
                }
            }, f, indent=2)
        
        return f"Exported {len(puzzles)} weakness puzzles to {output_file}"
    
    def print_analysis_report(self, results: Dict, report: Dict):
        """Print comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("🎮 V7P3R GAME REPLAY ANALYSIS REPORT")
        print("=" * 80)
        
        # Summary statistics
        summary = results["summary"]
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"Games processed: {summary['games_processed']}")
        print(f"Moves analyzed: {summary['moves_analyzed']}")
        print(f"Weaknesses found: {summary['weaknesses_found']}")
        print(f"Weakness rate: {summary['weakness_rate']:.1f}% of moves")
        print(f"Analysis time: {summary['analysis_time_minutes']:.1f} minutes")
        print(f"Processing rate: {summary['games_per_minute']:.1f} games/minute")
        
        if "error" in report:
            print(f"\n❌ Error generating report: {report['error']}")
            return
        
        # Weakness summary
        weakness_summary = report["weakness_summary"]
        print(f"\n🎯 WEAKNESS ANALYSIS:")
        print(f"Total weakness positions: {weakness_summary['total_weaknesses']}")
        print(f"Average centipawn loss: {weakness_summary['avg_centipawn_loss']:.0f}")
        print(f"Worst centipawn loss: {weakness_summary['worst_centipawn_loss']:.0f}")
        print(f"Time pressure factor: {weakness_summary['time_pressure_rate']:.1f}% of mistakes under time pressure")
        print(f"Current engine improvement: {weakness_summary['current_improvement_rate']:.1f}% of positions now handled better")
        
        # Theme analysis
        print(f"\n🎨 WEAKNESS THEMES (by severity):")
        for theme, stats in list(report["theme_analysis"].items())[:10]:
            print(f"  {theme:<20}: {stats['count']:3d} occurrences, "
                  f"avg {stats['avg_centipawn_loss']:3.0f}cp loss, "
                  f"worst {stats['worst_centipawn_loss']:3.0f}cp")
        
        # Phase analysis
        print(f"\n♟️  GAME PHASE PERFORMANCE:")
        for phase, stats in report["phase_analysis"].items():
            print(f"  {phase.title():<12}: {stats['count']:3d} mistakes, "
                  f"avg {stats['avg_loss']:3.0f}cp loss, "
                  f"worst {stats['worst_loss']:3.0f}cp")
        
        # Worst positions
        print(f"\n💥 TOP 5 WORST POSITIONS:")
        for i, pos in enumerate(report["worst_positions"][:5], 1):
            print(f"  {i}. Game {pos['game_id']}, move {pos['move_number']}")
            print(f"     V7P3R: {pos['v7p3r_move']} | Stockfish: {pos['stockfish_best']} | Loss: {pos['centipawn_loss']}cp")
            print(f"     FEN: {pos['fen']}")
            print(f"     Themes: {', '.join(pos['themes'])}")
        
        # Recommendations
        print(f"\n💡 IMPROVEMENT RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        print("=" * 80)
    
    def save_analysis_results(self, results: Dict, report: Dict, output_file: Optional[str] = None):
        """Save complete analysis results to JSON file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"v7p3r_game_analysis_{timestamp}.json"
        
        # Convert weakness positions to serializable format
        serializable_weaknesses = []
        for weakness in self.weakness_positions:
            weakness_dict = {
                "game_id": weakness.game_id,
                "move_number": weakness.move_number,
                "fen_before_move": weakness.fen_before_move,
                "opponent_last_move": weakness.opponent_last_move,
                "v7p3r_move": weakness.v7p3r_move,
                "v7p3r_move_rank": weakness.v7p3r_move_rank,
                "stockfish_top_moves": weakness.stockfish_top_moves,
                "stockfish_best_move": weakness.stockfish_best_move,
                "centipawn_loss": weakness.centipawn_loss,
                "position_themes": weakness.position_themes,
                "time_pressure": weakness.time_pressure,
                "material_balance": weakness.material_balance,
                "game_phase": weakness.game_phase
            }
            
            # Add current analysis if available
            if hasattr(weakness, 'current_analysis'):
                weakness_dict["current_analysis"] = weakness.current_analysis
            
            serializable_weaknesses.append(weakness_dict)
        
        complete_results = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "stockfish_path": self.stockfish_path,
                "v7p3r_engine_path": self.v7p3r_engine_path,
                "analysis_version": "1.0"
            },
            "processing_results": results,
            "weakness_analysis_report": report,
            "weakness_positions": serializable_weaknesses
        }
        
        with open(output_file, 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        print(f"\n💾 Complete analysis saved to: {output_file}")
        return output_file


def main():
    """Main execution function"""
    print("🎮 V7P3R Game Replay Analyzer")
    print("Analyzes historical games to identify performance blind spots")
    print("=" * 60)
    
    # Default configuration
    game_directory = r"S:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot"
    player_name = "v7p3r_bot"
    max_games = 50  # Limit for testing
    
    # Alternative: process command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='V7P3R Game Replay Analyzer')
    parser.add_argument('--directory', default=game_directory, help='Directory containing PGN files')
    parser.add_argument('--player', default=player_name, help='Player name to analyze')
    parser.add_argument('--max-games', type=int, default=max_games, help='Maximum games to process')
    parser.add_argument('--no-move-analysis', action='store_true', help='Skip current move ordering analysis')
    parser.add_argument('--export-puzzles', type=str, help='Export weakness puzzles to file')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = V7P3RGameReplayAnalyzer()
        
        # Process games
        print(f"Starting analysis of {args.player} games...")
        results = analyzer.process_game_directory(
            directory=args.directory,
            player_name=args.player,
            max_games=args.max_games,
            analyze_move_ordering=not args.no_move_analysis
        )
        
        if not results:
            print("No results generated. Check directory path and player name.")
            return 1
        
        # Generate weakness report
        print("\nGenerating weakness analysis report...")
        report = analyzer.generate_weakness_report(results)
        
        # Print report
        analyzer.print_analysis_report(results, report)
        
        # Save results
        output_file = analyzer.save_analysis_results(results, report)
        
        # Export weakness puzzles if requested
        if args.export_puzzles:
            puzzle_result = analyzer.export_weakness_puzzles(args.export_puzzles)
            print(f"\n🧩 {puzzle_result}")
        
        print(f"\n✅ Analysis complete! Results saved to {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())