# engine_utilities/stockfish_handler.py

import subprocess
import threading
import queue
import chess
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import logging
from typing import Optional, Dict, Any, Callable # Added Callable import

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base = getattr(sys, '_MEIPASS', None)
    if base:
        return os.path.join(base, relative_path)
    return os.path.join(os.path.abspath(".."), relative_path)

# At module level, define a single logger for this file
stockfish_handler_logger = logging.getLogger("stockfish_handler")
stockfish_handler_logger.setLevel(logging.DEBUG)
if not stockfish_handler_logger.handlers:
    if not os.path.exists('logging'):
        os.makedirs('logging', exist_ok=True)
    from logging.handlers import RotatingFileHandler
    log_file_path = "logging/stockfish_handler.log"
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10*1024*1024,
        backupCount=3,
        delay=True
    )
    formatter = logging.Formatter(
        '%(asctime)s | %(funcName)-15s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    stockfish_handler_logger.addHandler(file_handler)
    stockfish_handler_logger.propagate = False



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
        self.logger = stockfish_handler_logger

        if self.debug_mode:
            self.logger.debug(f"Initializing StockfishHandler from {self.stockfish_path}")
        else:
            self.logger.disabled = True

        self._start_engine() # This method attempts to set self.process

        # Ensure that if _start_engine failed, subsequent calls don't crash
        if self.process is None:
            self.logger.critical("Stockfish engine failed to start during initialization. Functionality will be limited.")

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

            self.process = subprocess.Popen(
                resource_path(self.stockfish_path),
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
            self.logger.info("Stockfish engine started successfully.")
            
            self._send_command("uci")
            self._wait_for_response("uciok")
            self._set_options()
            self._send_command("isready")
            self._wait_for_response("readyok")
            self.logger.info("Stockfish engine ready.")
        except FileNotFoundError:
            self.logger.error(f"Stockfish executable not found at: {self.stockfish_path}")
            self.process = None # Ensure process is None on failure
            raise # Re-raise to indicate critical failure
        except IOError as e:
            self.logger.error(f"IOError with Stockfish process pipes: {e}")
            self.process = None
            raise
        except Exception as e:
            self.logger.error(f"Failed to start Stockfish engine: {e}")
            self.process = None # Ensure process is None on failure
            raise

    def _enqueue_stdout(self):
        """Reads stdout from the Stockfish process and puts lines into a queue."""
        # Check that self.process and self.process.stdout are not None before iterating
        if self.process and self.process.stdout:
            for line in iter(self.process.stdout.readline, ''): # Pylance error fix: check self.process.stdout
                self.stdout_queue.put(line)
            self.process.stdout.close() # Pylance error fix: check self.process.stdout
        else:
            self.logger.error("Attempted to enqueue stdout but Stockfish process or stdout pipe is None.")


    def _send_command(self, command: str):
        """Sends a command to the Stockfish engine."""
        # Pylance error fix: Check self.process and self.process.stdin explicitly
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()
                if self.debug_mode:
                    self.logger.debug(f"Sent: {command}")
            except BrokenPipeError:
                self.logger.error(f"Failed to send command '{command}': Pipe to Stockfish is broken. Engine might have crashed.")
                self.process = None # Mark process as broken
            except Exception as e:
                self.logger.error(f"Error sending command '{command}' to Stockfish: {e}")
        else:
            self.logger.error("Stockfish process not running or stdin not available. Command not sent: %s", command)

    def _read_response(self, timeout: float = 5.0) -> str:
        """Reads a single line response from the Stockfish engine with a timeout."""
        try:
            line = self.stdout_queue.get(timeout=timeout)
            if self.debug_mode:
                self.logger.debug(f"Received: {line.strip()}")
            return line
        except queue.Empty:
            # This is a common case if engine is thinking or done. Not necessarily an error.
            return ""
        except Exception as e:
            self.logger.error(f"Error reading response from Stockfish: {e}")
            return ""

    def _wait_for_response(self, expected_end_string: str, timeout: float = 10.0) -> str:
        """Reads responses until a specific string is encountered or timeout."""
        if not self.process: # Ensure process exists before trying to read from it
            self.logger.error(f"Cannot wait for response '{expected_end_string}': Stockfish process is not running.")
            return ""

        full_response = []
        start_time = time.time()
        while time.time() - start_time < timeout:
            line = self._read_response(timeout=timeout - (time.time() - start_time))
            if not line:
                # If no line received for a period, check if process is still alive
                if self.process.poll() is not None: # Process has terminated
                    self.logger.error(f"Stockfish process terminated unexpectedly while waiting for '{expected_end_string}'.")
                    self.process = None
                    break
                continue # Keep waiting if process is alive but silent

            full_response.append(line)
            if expected_end_string in line:
                break
        else: # This else block executes if the while loop completes without a 'break' (i.e., it timed out)
            self.logger.warning(f"Timed out waiting for Stockfish response '{expected_end_string}'. Current response: {''.join(full_response).strip()[:100]}...")
            if self.process.poll() is not None: # Check if process died during timeout
                self.logger.error(f"Stockfish process terminated during timeout while waiting for '{expected_end_string}'.")
                self.process = None

        return "".join(full_response)

    def _set_options(self):
        """Sets Stockfish engine options (ELO, Skill Level)."""
        if self.process is None:
            self.logger.warning("Stockfish process not running, cannot set options.")
            return

        if self.elo_rating is not None:
            self._send_command(f"setoption name UCI_LimitStrength value true")
            self._send_command(f"setoption name UCI_Elo value {self.elo_rating}")
            self.logger.info(f"Set Stockfish ELO rating to {self.elo_rating}")
        if self.skill_level is not None:
            self._send_command(f"setoption name Skill Level value {self.skill_level}")
            self._send_command(f"setoption name UCI_LimitStrength value true") # Enable limiting for skill level
            self.logger.info(f"Set Stockfish Skill Level to {self.skill_level}")
        self._send_command(f"setoption name Use NNUE value true")

    def set_position(self, board: chess.Board):
        """Sets the current board position for the engine."""
        if self.process is None:
            self.logger.warning("Stockfish process not running, cannot set position.")
            return
        self._send_command(f"position fen {board.fen()}")
        self.logger.debug(f"Set position: {board.fen()}")

    def search(self, board: chess.Board, player: chess.Color, engine_config: Dict[str, Any], stop_callback: Optional[Callable[[], bool]] = None):
        """
        Initiates a search for the best move for the current position.
        This function will parse Stockfish's 'info' lines for nodes searched and best move.
        """
        if self.process is None:
            self.logger.error("Stockfish process not running, cannot perform search. Returning null move.")
            return chess.Move.null()

        self.set_position(board)
        self.nodes_searched = 0
        self.last_search_info = {'score': 0.0, 'nodes': 0, 'pv': ''}

        move_time_limit_ms = engine_config.get('time_limit', 0)
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
                self.logger.info("Stop callback triggered, attempting to stop Stockfish search.")
                self._send_command("stop") # Tell Stockfish to stop its current search
                # Wait for bestmove or a short period
                line = self._read_response(timeout=0.5)
                if line.startswith("bestmove"):
                    best_move_uci = line.split()[1]
                break # Exit the loop

            line = self._read_response(timeout=0.1)
            if not line:
                # If no line for a short period, and we haven't received bestmove, check if process died
                if self.process.poll() is not None:
                    self.logger.error("Stockfish process terminated during search. Cannot get bestmove.")
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
                if self.debug_mode:
                    self.logger.debug(f"Stockfish bestmove: {best_move_uci}")
                break
        
        if best_move_uci:
            try:
                move = chess.Move.from_uci(best_move_uci)
                return move
            except ValueError:
                self.logger.error(f"Stockfish returned invalid UCI move: {best_move_uci}. Returning null move.")
                return chess.Move.null()
        
        self.logger.error("Stockfish did not return a valid bestmove. Returning null move.")
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
                self.logger.warning(f"Could not parse score from info line: {line}")

        if "nodes" in parts:
            try:
                nodes_index = parts.index("nodes")
                self.last_search_info['nodes'] = int(parts[nodes_index + 1])
                self.nodes_searched = self.last_search_info['nodes']
            except (ValueError, IndexError):
                self.logger.warning(f"Could not parse nodes from info line: {line}")

        if "pv" in parts:
            try:
                pv_index = parts.index("pv")
                self.last_search_info['pv'] = " ".join(parts[pv_index + 1:])
            except (ValueError, IndexError):
                self.logger.warning(f"Could not parse PV from info line: {line}")


    def get_last_search_info(self) -> Dict[str, Any]:
        """Returns the last parsed search information (score, nodes, pv)."""
        return self.last_search_info


    def evaluate_position(self, board: chess.Board):
        """
        Gets the static evaluation of a position from Stockfish without searching for a move.
        This is typically done by sending 'go depth 0' or similar.
        """
        if self.process is None:
            self.logger.warning("Stockfish process not running, cannot evaluate position. Returning 0.0.")
            return 0.0

        self.set_position(board)
        self._send_command("go depth 0")
        
        score_cp = 0.0 # Initialize as float
        best_move_found = False
        
        start_time = time.time()
        while time.time() - start_time < 2.0: # Give it up to 2 seconds to respond
            line = self._read_response(timeout=0.1)
            if not line:
                # If no line received for a short period, check if process is still alive
                if self.process.poll() is not None:
                    self.logger.error("Stockfish process terminated during evaluation. Cannot get evaluation.")
                    self.process = None
                    break
                continue
            
            if line.startswith("info"):
                self._parse_info_line(line)
                score_cp = self.last_search_info.get('score', 0.0)
            elif line.startswith("bestmove"):
                best_move_found = True
                break

        if not best_move_found:
             self.logger.warning("Stockfish did not return 'bestmove' after 'go depth 0'. Using last info score.")

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
            self.close()
        except Exception:
            pass

    def quit(self):
        """Quits the Stockfish engine process."""
        if self.process and self.process.poll() is None:  # Check if process is still running
            try:
                self._send_command("quit")
                self.process.wait(timeout=3)  # Increased timeout slightly
                self.logger.info("Stockfish engine quit successfully.")
            except subprocess.TimeoutExpired:
                self.logger.warning("Stockfish engine did not quit gracefully, terminating...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.logger.warning("Stockfish engine did not terminate, killing process...")
                    self.process.kill()
                    self.process.wait()  # Wait for kill to complete
            except Exception as e:
                self.logger.error(f"Error while quitting Stockfish: {e}")
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
            self.logger.info("StockfishHandler reset for new game.")
        else:
            self.logger.warning("Stockfish process not running, cannot reset. Attempting to restart engine.")
            try:
                self._start_engine() # Try to restart if it crashed
                if self.process: # If restart was successful
                    self.set_position(board if board else chess.Board())
                    self.nodes_searched = 0
                    self.last_search_info = {'score': 0.0, 'nodes': 0, 'pv': ''}
                    self.logger.info("StockfishHandler successfully restarted and reset.")
            except Exception as e:
                self.logger.error(f"Failed to reset and restart StockfishHandler: {e}")
