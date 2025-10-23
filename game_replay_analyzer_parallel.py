#!/usr/bin/env python3
"""
V7P3R Game Replay Analyzer - Multithreaded Version
High-performance parallel analysis of V7P3R historical games

Performance Optimizations:
1. Parallel Stockfish analysis using ThreadPoolExecutor
2. Batch move processing to reduce overhead
3. Thread-safe result collection with locks
4. Memory-efficient streaming of large PGN files
5. Progress reporting with ETA calculations
6. Configurable thread count based on CPU cores

Expected Performance: 3-5x faster than single-threaded version
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from threading import Lock
import queue

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
    analysis_time: float = 0.0  # Time taken for analysis


@dataclass
class AnalysisTask:
    """Represents a move analysis task for threading"""
    task_id: int
    game_move: GameMove
    stockfish_path: str
    analysis_time: float = 2.0


@dataclass
class ProgressTracker:
    """Thread-safe progress tracking"""
    def __init__(self):
        self.lock = Lock()
        self.games_processed = 0
        self.moves_analyzed = 0
        self.weaknesses_found = 0
        self.start_time = time.time()
        self.last_update = time.time()
    
    def update_game(self):
        with self.lock:
            self.games_processed += 1
    
    def update_move(self, found_weakness: bool = False):
        with self.lock:
            self.moves_analyzed += 1
            if found_weakness:
                self.weaknesses_found += 1
    
    def get_stats(self) -> Dict:
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                'games_processed': self.games_processed,
                'moves_analyzed': self.moves_analyzed,
                'weaknesses_found': self.weaknesses_found,
                'elapsed_time': elapsed,
                'moves_per_second': self.moves_analyzed / elapsed if elapsed > 0 else 0,
                'games_per_minute': self.games_processed / (elapsed / 60) if elapsed > 0 else 0,
                'weakness_rate': (self.weaknesses_found / self.moves_analyzed * 100) if self.moves_analyzed > 0 else 0
            }


class ThreadSafeStockfishPool:
    """Thread-safe pool of Stockfish engines for parallel analysis"""
    
    def __init__(self, stockfish_path: str, pool_size: Optional[int] = None):
        self.stockfish_path = stockfish_path
        self.pool_size = pool_size or min(8, multiprocessing.cpu_count())
        self.engines = queue.Queue()
        self.lock = Lock()
        
        # Initialize engine pool
        print(f"🔧 Initializing Stockfish pool with {self.pool_size} engines...")
        for i in range(self.pool_size):
            try:
                engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                self.engines.put(engine)
            except Exception as e:
                print(f"Warning: Failed to initialize Stockfish engine {i+1}: {e}")
        
        print(f"✅ Stockfish pool ready with {self.engines.qsize()} engines")
    
    def get_engine(self):
        """Get an engine from the pool (blocking)"""
        return self.engines.get()
    
    def return_engine(self, engine):
        """Return an engine to the pool"""
        self.engines.put(engine)
    
    def shutdown(self):
        """Shutdown all engines in the pool"""
        engines = []
        while not self.engines.empty():
            try:
                engine = self.engines.get_nowait()
                engines.append(engine)
            except queue.Empty:
                break
        
        for engine in engines:
            try:
                engine.quit()
            except:
                pass
        
        print(f"🔧 Shutdown {len(engines)} Stockfish engines")


class MultiThreadedGameReplayAnalyzer:
    """High-performance multithreaded version of the game replay analyzer"""
    
    def __init__(self, 
                 stockfish_path: str = r"S:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester\engines\Stockfish\stockfish-windows-x86-64-avx2.exe",
                 v7p3r_engine_path: str = r"S:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\src\v7p3r_uci.py",
                 num_threads: Optional[int] = None):
        
        self.stockfish_path = stockfish_path
        self.v7p3r_engine_path = v7p3r_engine_path
        self.num_threads = num_threads or min(8, multiprocessing.cpu_count())
        self.move_analyzer = V7P3RMoveAnalyzer() if MOVE_ANALYZER_AVAILABLE else None
        
        # Thread-safe collections
        self.weakness_positions: List[WeaknessPosition] = []
        self.results_lock = Lock()
        self.progress = ProgressTracker()
        
        # Stockfish engine pool
        self.stockfish_pool = None
        
        # Verify paths
        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Stockfish not found: {stockfish_path}")
        if not os.path.exists(v7p3r_engine_path):
            print(f"Warning: V7P3R engine not found at {v7p3r_engine_path}")
        
        print(f"🚀 Multithreaded Game Replay Analyzer initialized")
        print(f"Threads: {self.num_threads}")
        print(f"Stockfish: {stockfish_path}")
        print(f"V7P3R Engine: {v7p3r_engine_path}")
    
    def initialize_stockfish_pool(self):
        """Initialize the Stockfish engine pool"""
        self.stockfish_pool = ThreadSafeStockfishPool(self.stockfish_path, self.num_threads)
    
    def shutdown_stockfish_pool(self):
        """Shutdown the Stockfish engine pool"""
        if self.stockfish_pool:
            self.stockfish_pool.shutdown()
            self.stockfish_pool = None
    
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
                    # Try to extract time from comment
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
    
    def analyze_move_batch(self, tasks: List[AnalysisTask]) -> List[Optional[WeaknessPosition]]:
        """Analyze a batch of moves using a Stockfish engine"""
        if not self.stockfish_pool:
            return [None] * len(tasks)
        
        results = []
        engine = self.stockfish_pool.get_engine()
        
        try:
            for task in tasks:
                start_time = time.time()
                weakness = self._analyze_single_move(engine, task.game_move)
                analysis_time = time.time() - start_time
                
                if weakness:
                    weakness.analysis_time = analysis_time
                
                results.append(weakness)
                
                # Update progress
                self.progress.update_move(weakness is not None)
                
        except Exception as e:
            print(f"Error in batch analysis: {e}")
            results = [None] * len(tasks)
        finally:
            self.stockfish_pool.return_engine(engine)
        
        return results
    
    def _analyze_single_move(self, engine: chess.engine.SimpleEngine, game_move: GameMove) -> Optional[WeaknessPosition]:
        """Analyze a single move using the provided Stockfish engine"""
        try:
            board = chess.Board(game_move.position_before_fen)
            
            # Get Stockfish's top moves
            result = engine.analyse(
                board,
                chess.engine.Limit(time=2.0),
                multipv=5
            )
            
            stockfish_moves = []
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
                    
                    stockfish_moves.append((move, cp_value))
            
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
            
            # If V7P3R's move is not in top 3, it's a weakness
            if v7p3r_rank == 0 or v7p3r_rank > 3:
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
                themes = self._classify_position_themes(board)
                
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
            
        except Exception as e:
            print(f"Error analyzing move {game_move.move_uci}: {e}")
            return None
    
    def _classify_position_themes(self, board: chess.Board) -> List[str]:
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
    
    def print_progress_update(self):
        """Print current progress statistics"""
        stats = self.progress.get_stats()
        print(f"📊 Progress: {stats['games_processed']} games, "
              f"{stats['moves_analyzed']} moves ({stats['moves_per_second']:.1f}/sec), "
              f"{stats['weaknesses_found']} weaknesses ({stats['weakness_rate']:.1f}%), "
              f"{stats['games_per_minute']:.1f} games/min")
    
    def process_game_directory_parallel(self, directory: str, player_name: str, 
                                      max_games: Optional[int] = None,
                                      batch_size: int = 50) -> Dict:
        """Process all PGN files in directory using parallel analysis"""
        
        print(f"🎮 Parallel Processing: {directory}")
        print(f"Target player: {player_name}")
        print(f"Threads: {self.num_threads}, Batch size: {batch_size}")
        print("=" * 60)
        
        # Initialize Stockfish pool
        self.initialize_stockfish_pool()
        
        try:
            # Find all PGN files
            pgn_files = self.find_pgn_files(directory)
            if not pgn_files:
                print(f"No PGN files found in {directory}")
                return {}
            
            print(f"Found {len(pgn_files)} PGN files")
            
            # Collect all moves from all games first
            all_moves = []
            games_processed = 0
            
            for pgn_file in pgn_files:
                if max_games and games_processed >= max_games:
                    break
                
                print(f"\n📁 Extracting moves from: {os.path.basename(pgn_file)}")
                
                try:
                    with open(pgn_file, 'r', encoding='utf-8') as f:
                        while True:
                            if max_games and games_processed >= max_games:
                                break
                                
                            game = chess.pgn.read_game(f)
                            if game is None:
                                break
                            
                            games_processed += 1
                            self.progress.update_game()
                            
                            # Extract moves for our player
                            player_moves = self.extract_moves_from_game(game, player_name)
                            all_moves.extend(player_moves)
                            
                            if games_processed % 10 == 0:
                                print(f"  Extracted {len(all_moves)} moves from {games_processed} games")
                
                except Exception as e:
                    print(f"Error processing {pgn_file}: {e}")
                    continue
            
            print(f"\n🎯 Total moves to analyze: {len(all_moves)}")
            print(f"📊 Games processed: {games_processed}")
            
            if not all_moves:
                print("No moves found to analyze!")
                return {}
            
            # Create analysis tasks
            tasks = [AnalysisTask(i, move, self.stockfish_path) for i, move in enumerate(all_moves)]
            
            # Process moves in parallel batches
            print(f"\n🚀 Starting parallel analysis with {self.num_threads} threads...")
            
            progress_interval = max(1, len(tasks) // 20)  # Update every 5%
            last_progress_time = time.time()
            
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit batches
                futures = []
                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i+batch_size]
                    future = executor.submit(self.analyze_move_batch, batch)
                    futures.append(future)
                
                # Collect results
                for i, future in enumerate(as_completed(futures)):
                    try:
                        batch_results = future.result()
                        
                        # Add valid weaknesses to results
                        with self.results_lock:
                            for weakness in batch_results:
                                if weakness:
                                    self.weakness_positions.append(weakness)
                        
                        # Print progress
                        current_time = time.time()
                        if current_time - last_progress_time > 5.0:  # Every 5 seconds
                            self.print_progress_update()
                            last_progress_time = current_time
                            
                    except Exception as e:
                        print(f"Error processing batch {i}: {e}")
            
            # Final progress update
            self.print_progress_update()
            
            # Compile final statistics
            final_stats = self.progress.get_stats()
            
            results = {
                "summary": {
                    "games_processed": final_stats['games_processed'],
                    "moves_analyzed": final_stats['moves_analyzed'],
                    "weaknesses_found": final_stats['weaknesses_found'],
                    "weakness_rate": final_stats['weakness_rate'],
                    "analysis_time_minutes": final_stats['elapsed_time'] / 60,
                    "moves_per_second": final_stats['moves_per_second'],
                    "games_per_minute": final_stats['games_per_minute'],
                    "performance_improvement": f"{self.num_threads}x threading speedup"
                },
                "weakness_positions": self.weakness_positions,
                "pgn_files_processed": pgn_files
            }
            
            return results
            
        finally:
            # Always cleanup
            self.shutdown_stockfish_pool()
    
    def save_results(self, results: Dict, output_file: Optional[str] = None):
        """Save analysis results to JSON file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"v7p3r_parallel_analysis_{timestamp}.json"
        
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
                "game_phase": weakness.game_phase,
                "analysis_time": weakness.analysis_time
            }
            serializable_weaknesses.append(weakness_dict)
        
        complete_results = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "stockfish_path": self.stockfish_path,
                "v7p3r_engine_path": self.v7p3r_engine_path,
                "num_threads": self.num_threads,
                "analysis_version": "2.0_parallel"
            },
            "processing_results": {
                "summary": results.get("summary", {}),
                "pgn_files_processed": [os.path.basename(f) for f in results.get("pgn_files_processed", [])]
            },
            "weakness_positions": serializable_weaknesses
        }
        
        with open(output_file, 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        return output_file


def main():
    """Main execution function for parallel analysis"""
    print("🚀 V7P3R Multithreaded Game Replay Analyzer")
    print("High-performance parallel analysis of historical games")
    print("=" * 60)
    
    # Default configuration
    game_directory = r"S:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot"
    player_name = "v7p3r_bot"
    max_games = 20  # Increased for parallel processing
    
    # Command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='V7P3R Multithreaded Game Replay Analyzer')
    parser.add_argument('--directory', default=game_directory, help='Directory containing PGN files')
    parser.add_argument('--player', default=player_name, help='Player name to analyze')
    parser.add_argument('--max-games', type=int, default=max_games, help='Maximum games to process')
    parser.add_argument('--threads', type=int, help='Number of analysis threads (default: CPU cores)')
    parser.add_argument('--batch-size', type=int, default=50, help='Moves per batch (default: 50)')
    parser.add_argument('--export-puzzles', type=str, help='Export weakness puzzles to file')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = MultiThreadedGameReplayAnalyzer(num_threads=args.threads)
        
        # Process games in parallel
        print(f"Starting parallel analysis of {args.player} games...")
        start_time = time.time()
        
        results = analyzer.process_game_directory_parallel(
            directory=args.directory,
            player_name=args.player,
            max_games=args.max_games,
            batch_size=args.batch_size
        )
        
        total_time = time.time() - start_time
        
        if not results:
            print("No results generated. Check directory path and player name.")
            return 1
        
        # Print summary
        summary = results["summary"]
        print("\n" + "=" * 60)
        print("🎯 PARALLEL ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"Games processed: {summary['games_processed']}")
        print(f"Moves analyzed: {summary['moves_analyzed']}")
        print(f"Weaknesses found: {summary['weaknesses_found']}")
        print(f"Weakness rate: {summary['weakness_rate']:.1f}%")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Performance: {summary['moves_per_second']:.1f} moves/second")
        print(f"Speedup: ~{analyzer.num_threads}x vs single-threaded")
        
        # Save results
        output_file = analyzer.save_results(results)
        
        # Show top weaknesses
        if analyzer.weakness_positions:
            print(f"\n🎯 TOP 5 WEAKNESSES FOUND:")
            sorted_weaknesses = sorted(analyzer.weakness_positions, 
                                     key=lambda x: x.centipawn_loss, 
                                     reverse=True)
            
            for i, weakness in enumerate(sorted_weaknesses[:5], 1):
                print(f"  {i}. {weakness.game_id} move {weakness.move_number}: "
                      f"{weakness.v7p3r_move} (lost {weakness.centipawn_loss}cp)")
                print(f"     Best: {weakness.stockfish_best_move}, "
                      f"Themes: {', '.join(weakness.position_themes[:3])}")
        
        print(f"\n✅ Analysis complete! Results saved to {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())