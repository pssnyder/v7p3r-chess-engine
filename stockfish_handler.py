# stockfish_handler.py

import subprocess
import threading
import queue
import chess
import time
import sys
import os
from typing import Optional, Dict, Any, Callable
from v7p3r_paths import paths

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class StockfishHandler:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, stockfish_config):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StockfishHandler, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, stockfish_config):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self.stockfish_path = stockfish_config.get('stockfish_path')
        self.elo_rating = stockfish_config.get('elo_rating')
        self.skill_level = stockfish_config.get('skill_level')
        self.process = None # Initialize as None
        self.stdout_queue = queue.Queue()
        self.stdout_thread = None
        self.debug_mode = stockfish_config.get('debug_mode', False)

        self._start_engine() # This method attempts to set self.process

        self.nodes_searched = 0
        self.last_search_info = {}

    def _start_engine(self):
        """Starts the Stockfish engine process."""
        try:
            # Using Popen with PIPE for stdin/stdout to communicate
            # Added `creationflags` for Windows to hide the console window.
            # This is a common practice for background processes.
            creationflags = 0
            if os.name == 'nt': # If Windows
                creationflags = subprocess.CREATE_NO_WINDOW

            stockfish_executable = str(paths.get_resource_path(self.stockfish_path))
            
            # Try to verify Stockfish executable is valid
            if not os.path.exists(stockfish_executable):
                raise FileNotFoundError(f"Stockfish executable not found at: {stockfish_executable}")
            
            self.process = subprocess.Popen(
                stockfish_executable,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creationflags # Apply creation flags
            )
            
            # Check if stdout/stdin pipes are valid before starting thread/sending commands
            if self.process.stdout is None or self.process.stdin is None:
                raise IOError("Failed to open stdout/stdin pipes for Stockfish process.")

            # Start a thread to read stdout without blocking
            self.stdout_thread = threading.Thread(target=self._enqueue_stdout, daemon=True)
            self.stdout_thread.start()
            
            # Quick echo test
            self._send_command("echo test_string")
            response = self._wait_for_response("test_string", timeout=2.0)
            if not response:
                raise IOError("Stockfish failed echo test - process may not be responding")
            
            # Initialize UCI mode
            self._send_command("uci")
            self._set_options()
            self._send_command("isready")
            
        except Exception as e:
            self.process = None
            raise

    def _enqueue_stdout(self):
        """Reads stdout from the Stockfish process and puts lines into a queue."""
        # Check that self.process and self.process.stdout are not None before iterating
        if self.process and self.process.stdout:
            for line in iter(self.process.stdout.readline, ''): # Pylance error fix: check self.process.stdout
                self.stdout_queue.put(line)
            self.process.stdout.close() # Pylance error fix: check self.process.stdout

    def _send_command(self, command: str):
        """Sends a command to the Stockfish engine."""
        # Pylance error fix: Check self.process and self.process.stdin explicitly
        if self.process and self.process.stdin:
            # Check if process has terminated before sending
            if self.process and hasattr(self.process, 'poll'):
                poll_result = self.process.poll()
                if poll_result is not None:
                    self.process = None
                    return
                
            try:
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()
                
                # Check if process has terminated after sending
                if self.process and hasattr(self.process, 'poll'):
                    poll_result = self.process.poll()
                    if poll_result is not None:
                        self.process = None
                    return
                    
            except BrokenPipeError:
                self.process = None # Mark process as broken
            except Exception as e:
                raise

    def _read_response(self, timeout: float = 5.0) -> str:
        """Reads a single line response from the Stockfish engine with a timeout."""
        try:
            line = self.stdout_queue.get(timeout=timeout)
            return line
        except queue.Empty:
            # This is a common case if engine is thinking or done. Not necessarily an error.
            return ""
        except Exception as e:
            return ""

    def _wait_for_response(self, expected_end_string: str, timeout: float = 10.0) -> str:
        """Reads responses until a specific string is encountered or timeout."""
        if not self.process: # Ensure process exists before trying to read from it
            return ""

        full_response = []
        start_time = time.time()
        while time.time() - start_time < timeout:
            line = self._read_response(timeout=timeout - (time.time() - start_time))
            if not line:
                # If no line received for a period, check if process is still alive
                if self.process.poll() is not None: # Process has terminated
                    self.process = None
                    break
                continue # Keep waiting if process is alive but silent

            full_response.append(line)
            if expected_end_string in line:
                break
        else: # This else block executes if the while loop completes without a 'break' (i.e., it timed out)
            if self.process and hasattr(self.process, 'poll'):
                poll_result = self.process.poll()
                if poll_result is not None: # Check if process died during timeout
                    self.process = None

        return "".join(full_response)

    def _set_options(self):
        """Sets Stockfish engine options (ELO, Skill Level)."""
        if self.process is None:
            return

        if self.elo_rating is not None:
            self._send_command(f"setoption name UCI_LimitStrength value true")
            self._send_command(f"setoption name UCI_Elo value {self.elo_rating}")
        if self.skill_level is not None:
            self._send_command(f"setoption name Skill Level value {self.skill_level}")
            self._send_command(f"setoption name UCI_LimitStrength value true") # Enable limiting for skill level
        # Note: Removed "Use NNUE" option as it's not supported in this Stockfish version

    def set_position(self, board: chess.Board):
        """Sets the current board position for the engine."""
        if self.process is None:
            return
        
        # Check if process has crashed
        if self.process and hasattr(self.process, 'poll') and self.process.poll() is not None:
            self.process = None
            return
            
        self._send_command(f"position fen {board.fen()}")
        
        # Check if process crashed after sending command
        if self.process and hasattr(self.process, 'poll') and self.process.poll() is not None:
            self.process = None
            return

    def search(self, board: chess.Board, player: chess.Color, engine_config: Dict[str, Any], stop_callback: Optional[Callable[[], bool]] = None):
        """
        Initiates a search for the best move for the current position.
        This function will parse Stockfish's 'info' lines for nodes searched and best move.
        """
        if self.process is None:
            # First try to restart the engine
            try:
                self._start_engine()
                self._set_options()
            except Exception as e:
                raise RuntimeError("Stockfish engine is not available") from e

        # Check if process is alive
        if self.process and hasattr(self.process, 'poll') and self.process.poll() is not None:
            self.process = None
            raise RuntimeError("Stockfish engine has crashed")
        
        # Try to set position
        try:
            self.set_position(board)
        except Exception as e:
            raise RuntimeError("Failed to set board position in Stockfish") from e
        
        self.nodes_searched = 0
        self.last_search_info = {'score': 0.0, 'nodes': 0, 'pv': ''}

        # Handle both 'time_limit' (expected) and 'movetime' (config) field names
        move_time_limit_ms = engine_config.get('time_limit', engine_config.get('movetime', 0))
        depth_limit = engine_config.get('depth', 0)

        command = "go"
        if move_time_limit_ms > 0:
            command += f" movetime {move_time_limit_ms}"
        elif depth_limit > 0:
            command += f" depth {depth_limit}"
        else:
            command += " movetime 1000" # Default to 1 second if no limits are specified

        self._send_command(command)

        best_move_uci = None
        
        while True:
            # Add a check for stop_callback, although Stockfish manages its own time/depth
            # This is more for external signals to stop the Python wrapper from waiting
            if stop_callback and stop_callback():
                self._send_command("stop") # Tell Stockfish to stop its current search
                # Wait for bestmove or a short period
                line = self._read_response(timeout=0.5)
                if line.startswith("bestmove"):
                    best_move_uci = line.split()[1]
                break # Exit the loop

            line = self._read_response(timeout=0.1)
            if not line:
                # If no line for a short period, and we haven't received bestmove, check if process died
                if self.process and hasattr(self.process, 'poll'):
                    poll_result = self.process.poll()
                    if poll_result is not None:
                        self.process = None
                        break
                if best_move_uci: # If we have a bestmove from a previous 'info' line (rare but possible)
                    break
                continue # Keep waiting for more output
            
            if line.startswith("info"):
                self._parse_info_line(line)
            elif line.startswith("bestmove"):
                parts = line.split()
                best_move_uci = parts[1]
                break
        
        if best_move_uci:
            try:
                move = chess.Move.from_uci(best_move_uci)
                return move
            except ValueError:
                return chess.Move.null()
        
        return chess.Move.null()

    def _parse_info_line(self, line: str):
        """Parses an 'info' line from Stockfish output to extract useful data."""
        parts = line.split()
        
        if "score" in parts:
            try:
                score_index = parts.index("score")
                score_type = parts[score_index + 1]
                score_value = int(parts[score_index + 2])
                
                if score_type == "cp":
                    self.last_search_info['score'] = score_value / 100.0
                elif score_type == "mate":
                    mate_sign = 1 if score_value > 0 else -1
                    self.last_search_info['score'] = mate_sign * (999999 - abs(score_value))
            except (ValueError, IndexError):
                raise

        if "nodes" in parts:
            try:
                nodes_index = parts.index("nodes")
                self.last_search_info['nodes'] = int(parts[nodes_index + 1])
                self.nodes_searched = self.last_search_info['nodes']
            except (ValueError, IndexError):
                raise

        if "pv" in parts:
            try:
                pv_index = parts.index("pv")
                self.last_search_info['pv'] = " ".join(parts[pv_index + 1:])
            except (ValueError, IndexError):
                raise


    def get_last_search_info(self) -> Dict[str, Any]:
        """Returns the last parsed search information (score, nodes, pv)."""
        return self.last_search_info


    def evaluate_position(self, board: chess.Board):
        """
        Gets the static evaluation of a position from Stockfish without searching for a move.
        This is typically done by sending 'go depth 0' or similar.
        """
        if self.process is None:
            return 0.0

        self.set_position(board)
        self._send_command("go depth 0")
        
        score_cp = 0.0 # Initialize as float
        best_move_found = False
        
        start_time = time.time()
        timeout = 1.0  # Reduced timeout to 1 second for faster GA training
        while time.time() - start_time < timeout:
            line = self._read_response(timeout=0.1)
            if not line:
                # If no line received for a short period, check if process is still alive
                if self.process.poll() is not None:
                    self.process = None
                    break
                continue
            
            if line.startswith("info"):
                self._parse_info_line(line)
                score_cp = self.last_search_info.get('score', 0.0)
            elif line.startswith("bestmove"):
                best_move_found = True
                break

        return score_cp

    def evaluate_position_from_perspective(self, board: chess.Board, player: chess.Color):
        """
        Evaluates the position from the perspective of the given player.
        Stockfish scores are usually from White's perspective.
        """
        score = self.evaluate_position(board)
        if player == chess.BLACK:
            return -score
        return score


    def close(self):
        """Close the Stockfish handler and terminate the process. Also resets singleton instance."""
        self.quit()
        with self._lock:
            type(self)._instance = None
            self._initialized = False

    def __del__(self):
        """Destructor to ensure Stockfish process is terminated when object is destroyed."""
        try:
            # Only quit the process, don't reset the singleton instance
            self.quit()
        except Exception:
            pass

    def quit(self):
        """Quits the Stockfish engine process."""
        if self.process and self.process.poll() is None:  # Check if process is still running
            try:
                self._send_command("quit")
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()  # Wait for kill to complete
            except Exception as e:
                # Force kill if there was an error
                try:
                    self.process.kill()
                    self.process.wait()
                except Exception:
                    pass
            finally:
                self.process = None  # Ensure process is None after attempting to quit
                if self.stdout_thread and self.stdout_thread.is_alive():
                    # Daemon thread will exit with main process
                    pass


    def reset(self, board: Optional[chess.Board] = None):
        """Resets the handler state, similar to how EvaluationEngine.reset works."""
        if self.process:
            self._send_command("ucinewgame")
            self._wait_for_response("readyok") # Wait for Stockfish to confirm reset
            self.set_position(board if board else chess.Board()) # Set initial position after reset
            self.nodes_searched = 0
            self.last_search_info = {'score': 0.0, 'nodes': 0, 'pv': ''}
        else:
            try:
                self._start_engine() # Try to restart if it crashed
                if self.process: # If restart was successful
                    self.set_position(board if board else chess.Board())
                    self.nodes_searched = 0
                    self.last_search_info = {'score': 0.0, 'nodes': 0, 'pv': ''}
            except Exception as e:
                raise
