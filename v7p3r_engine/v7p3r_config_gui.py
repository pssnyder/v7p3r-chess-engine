"""
V7P3R Chess Engine Configuration GUI

This module provides a graphical user interface for managing and executing chess engine configurations.
Users can:
- Load existing configurations from saved JSON files
- Edit configurations directly in JSON format
- Create new configurations using a guided form interface
- Save configurations for future use
- Run the chess game with the selected configuration

The configuration includes settings for both the V7P3R engine and opponent engines like Stockfish.
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import datetime
import yaml
from tkinter.font import Font
import copy
import logging

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base = getattr(sys, '_MEIPASS', None)
    if base:
        return os.path.join(base, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def get_log_file_path():
    # Optional timestamp for log file name
    timestamp = get_timestamp()
    log_dir = "logging"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"v7p3r_evaluation_engine.log")
v7p3r_engine_logger = logging.getLogger("v7p3r_evaluation_engine")
v7p3r_engine_logger.setLevel(logging.DEBUG)
_init_status = globals().get("_init_status", {})
if not _init_status.get("initialized", False):
    log_file_path = get_log_file_path()
    from logging.handlers import RotatingFileHandler
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
    v7p3r_engine_logger.addHandler(file_handler)
    v7p3r_engine_logger.propagate = False
    _init_status["initialized"] = True
    # Store the log file path for later use (e.g., to match with PGN/config)
    _init_status["log_file_path"] = log_file_path

# Import the ChessGame class from play_v7p3r
try:
    from v7p3r_engine.v7p3r_play import ChessGame  # Try local import first
except ImportError:
    from v7p3r_engine.v7p3r_play import ChessGame  # Fallback to package import

# Default values used for new configurations
DEFAULT_CONFIG = {
    "engine_config": {
        "name": "v7p3r",
        "version": "1.0.0",
        "color": "white",
        "ruleset": "default_evaluation",
        "search_algorithm": "lookahead",
        "depth": 2,
        "max_depth": 3,
        "monitoring_enabled": True,
        "verbose_output": False,
        "logger": "v7p3r_engine_logger",
        "game_count": 1,
        "starting_position": "default",
        "white_player": "v7p3r",
        "black_player": "stockfish"
    },
    "stockfish_config": {
        "stockfish_path": "engine_utilities/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe",
        "elo_rating": 400,
        "skill_level": 1,
        "debug_mode": False,
        "depth": 2,
        "max_depth": 2,
        "movetime": 1000
    }
}

# Path for saved configurations
CONFIG_DIR = os.path.join(parent_dir, "v7p3r_engine", "saved_configs")
os.makedirs(CONFIG_DIR, exist_ok=True)

# Path for rulesets file
RULESET_PATH = os.path.join(parent_dir, "v7p3r_engine", "rulesets", "rulesets.yaml")

# Check if rulesets.yaml exists, if not create it with default values
def ensure_rulesets_yaml_exists():
    """Ensure that rulesets.yaml exists with at least a default evaluation ruleset"""
    os.makedirs(os.path.dirname(RULESET_PATH), exist_ok=True)
    if not os.path.exists(RULESET_PATH):
        # Create with default ruleset
        default_ruleset = {
            "default_evaluation": get_default_ruleset_values()
        }
        try:
            with open(RULESET_PATH, 'w') as f:
                yaml.dump(default_ruleset, f, default_flow_style=False, sort_keys=False)
            print(f"Created default rulesets.yaml at {RULESET_PATH}")
        except Exception as e:
            print(f"Error creating default rulesets.yaml: {e}")

# Call to ensure the rulesets file exists
ensure_rulesets_yaml_exists()

# Load ruleset options from rulesets.yaml
def load_rulesets():
    try:
        with open(RULESET_PATH, 'r') as file:
            rulesets = yaml.safe_load(file)
            return list(rulesets.keys()) if rulesets else ["default_evaluation"]
    except Exception as e:
        print(f"Error loading rulesets: {e}")
        return ["default_evaluation"]
# Get available engine options
def get_engine_options():
    # Default engines always available
    engines = ["v7p3r", "stockfish", "human"]
    
    # Check for RL engine
    try:
        from v7p3r_rl_engine.v7p3r_rl import v7p3rRLEngine
        engines.append("v7p3r_rl")
    except ImportError:
        pass
    
    # Check for GA engine
    try:
        from v7p3r_ga_engine.v7p3r_ga import v7p3rGeneticAlgorithm
        engines.append("v7p3r_ga")
    except ImportError:
        pass
    
    # Check for NN engine
    try:
        from v7p3r_nn_engine.v7p3r_nn import v7p3rNeuralNetwork
        engines.append("v7p3r_nn")
    except ImportError:
        pass
    
    return engines

# Configuration management functions
def list_saved_configs():
    """List all saved configuration files"""
    if not os.path.exists(CONFIG_DIR):
        return []
    
    configs = []
    for filename in os.listdir(CONFIG_DIR):
        if filename.endswith('.json'):
            configs.append(filename[:-5])  # Remove .json extension
    return configs

def save_config(config_name, config_data):
    """Save configuration to a JSON file"""
    if not config_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = f"config_{timestamp}"
    
    # Ensure the config name has no spaces or special characters
    config_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in config_name)
    
    # Add .json extension if not present
    if not config_name.endswith('.json'):
        config_name += '.json'
    
    filepath = os.path.join(CONFIG_DIR, config_name)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=4)
        return True, config_name
    except Exception as e:
        return False, str(e)

def load_config(config_name):
    """Load configuration from a JSON file"""
    if not config_name.endswith('.json'):
        config_name += '.json'
    
    filepath = os.path.join(CONFIG_DIR, config_name)
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)

def validate_config(config_data):
    """Basic validation of configuration data"""
    required_keys = ["engine_config", "stockfish_config"]
    
    # Check if all required keys exist
    for key in required_keys:
        if key not in config_data:
            return False, f"Missing required key: {key}"
    
    # Check engine_config
    engine_config = config_data.get("engine_config", {})
    engine_required_keys = ["name", "version", "color", "ruleset", "search_algorithm", "depth", "max_depth", "monitoring_enabled", "logger", "game_count", "starting_position", "white_player", "black_player", "verbose_output"]
    for key in engine_required_keys:
        if key not in engine_config:
            return False, f"Missing required engine config key: {key}"
    
    # Check stockfish_config if black_player or white_player is stockfish
    if engine_config.get("white_player") == "stockfish" or engine_config.get("black_player") == "stockfish":
        stockfish_config = config_data.get("stockfish_config", {})
        stockfish_required_keys = ["stockfish_path", "elo_rating", "skill_level", "debug_mode", "depth", "max_depth", "movetime"]
        for key in stockfish_required_keys:
            if key not in stockfish_config:
                return False, f"Missing required stockfish config key: {key}"

    return True, None

# Ruleset management functions
def load_ruleset_data():
    """Load all rulesets from rulesets.yaml"""
    try:
        with open(RULESET_PATH, 'r') as file:
            rulesets = yaml.safe_load(file)
            return rulesets or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Error loading rulesets data: {e}")
        return {}

def save_ruleset(ruleset_name, ruleset_data, all_rulesets=None):
    """Save a ruleset to rulesets.yaml"""
    # If no name provided, generate timestamped name
    if not ruleset_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        ruleset_name = f"custom_evaluation_{timestamp}"
    
    # Ensure the ruleset name has no spaces or special characters
    ruleset_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in ruleset_name)
    
    try:
        # Load existing rulesets if not provided
        if all_rulesets is None:
            all_rulesets = load_ruleset_data()
        
        # Add or update the ruleset
        all_rulesets[ruleset_name] = ruleset_data
        
        # Save all rulesets back to file
        with open(RULESET_PATH, 'w') as f:
            yaml.dump(all_rulesets, f, default_flow_style=False, sort_keys=False)
        return True, ruleset_name
    except Exception as e:
        return False, str(e)

def get_default_ruleset_values():
    """Get default values for a new ruleset"""
    # Default values for a new ruleset
    return {
        "backward_pawn_penalty": -0.5,
        "bishop_activity_bonus": 0.15,
        "bishop_vision_bonus": 1.0,
        "capture_bonus": 15.0,
        "capture_move_bonus": 4000,
        "castling_bonus": 5.0,
        "castling_protection_bonus": 3.0,
        "castling_protection_penalty": -6.0,
        "center_control_bonus": 0.25,
        "check_bonus": 50.0,
        "check_move_bonus": 10000,
        "checkmate_bonus": 1000000.0,
        "checkmate_move_bonus": 1000000,
        "coordinated_rooks_bonus": 0.25,
        "counter_move_bonus": 1000,
        "doubled_pawn_penalty": -0.5,
        "draw_penalty": -9999999999.0,
        "en_passant_bonus": 1.0,
        "exposed_king_penalty": -10.0,
        "file_control_bonus": 0.2,
        "hanging_piece_bonus": 2.0,
        "hash_move_bonus": 5000,
        "history_move_bonus": 1000,
        "in_check_penalty": -10.0,
        "isolated_pawn_penalty": -0.5,
        "killer_move_bonus": 2000,
        "king_safety_bonus": 1.5,
        "king_safety_penalty": -100.0,
        "king_threat_penalty": -50.0,
        "knight_activity_bonus": 0.1,
        "knight_pair_bonus": 1.0,
        "knight_vision_penalty": -0.25,
        "material_weight": 0.8,
        "open_file_bonus": 0.3,
        "passed_pawn_bonus": 1.0,
        "pawn_advancement_bonus": 0.25,
        "pawn_promotion_bonus": 5.0,
        "pawn_structure_bonus": 0.1,
        "piece_activity_bonus": 0.1,
        "piece_coordination_bonus": 0.5,
        "piece_development_bonus": 2.0,
        "piece_mobility_bonus": 0.1,
        "promotion_move_bonus": 3000,
        "queen_capture_bonus": 1000.0,
        "repetition_penalty": -9999999999.0,
        "rook_development_penalty": 0.2,
        "rook_position_bonus": 0.4,
        "stacked_rooks_bonus": 0.5,
        "stalemate_penalty": -9999999999.0,
        "tempo_bonus": 0.1,
        "trapped_piece_penalty": -5.0,
        "undefended_piece_penalty": -2.0,
        "undeveloped_penalty": -0.5
    }

class ConfigGUI:
    """Main class for the V7P3R Configuration GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("V7P3R Chess Engine Configuration")
        self.root.geometry("900x800")
        self.root.minsize(800, 700)
        
        # Set a reasonable DPI-aware font size
        default_font = Font(family="Segoe UI", size=10)
        text_font = Font(family="Consolas", size=10)
        
        # Dark mode color scheme (accessible and easy to view)
        self.colors = {
            "bg": "#282c34",
            "fg": "#abb2bf",
            "button": "#3e4451",
            "button_fg": "#ffffff",
            "highlight": "#61afef",
            "error": "#e06c75",
            "success": "#98c379",
            "text_bg": "#21252b",
            "text_fg": "#d4d4d4",
            "header": "#c678dd"
        }
        
        # Apply colors to root window
        self.root.configure(bg=self.colors["bg"])
        
        # Create style for ttk widgets
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use a theme that supports custom colors
        
        # Configure ttk styles
        self.style.configure('TFrame', background=self.colors["bg"])
        self.style.configure('TLabel', background=self.colors["bg"], foreground=self.colors["fg"], font=default_font)
        self.style.configure('TButton', background=self.colors["button"], foreground=self.colors["button_fg"], font=default_font)
        self.style.map('TButton', 
                      background=[('active', self.colors["highlight"])],
                      foreground=[('active', self.colors["button_fg"])])
        self.style.configure('TEntry', foreground=self.colors["text_fg"], fieldbackground=self.colors["text_bg"], font=default_font)
        self.style.configure('TCombobox', foreground=self.colors["text_fg"], fieldbackground=self.colors["text_bg"], font=default_font)
        self.style.map('TCombobox', 
                      fieldbackground=[('readonly', self.colors["text_bg"])],
                      foreground=[('readonly', self.colors["text_fg"])])
        
        # Current configuration
        self.current_config = copy.deepcopy(DEFAULT_CONFIG)
        self.rulesets = load_rulesets()
        self.engine_options = get_engine_options()
        
        # Load ruleset data
        self.ruleset_data = load_ruleset_data()
        self.current_ruleset = get_default_ruleset_values()
        self.ruleset_vars = {}
        
        # Main container frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.load_tab = ttk.Frame(self.notebook, padding=10)
        self.json_tab = ttk.Frame(self.notebook, padding=10)
        self.form_tab = ttk.Frame(self.notebook, padding=10)
        self.ruleset_tab = ttk.Frame(self.notebook, padding=10)
        self.ruleset_editor_tab = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.load_tab, text="Load Configuration")
        self.notebook.add(self.json_tab, text="JSON Editor")
        self.notebook.add(self.form_tab, text="Guided Configuration")
        self.notebook.add(self.ruleset_tab, text="Ruleset Manager")
        self.notebook.add(self.ruleset_editor_tab, text="Ruleset Editor")
        
        # Setup each tab
        self._setup_load_tab()
        self._setup_json_tab()
        self._setup_form_tab()
        self._setup_ruleset_tab()
        self._setup_ruleset_editor_tab()
        
        # Bottom buttons frame for global actions
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Buttons at the bottom
        button_style = ttk.Style()
        button_style.configure('Action.TButton', font=('Segoe UI', 11, 'bold'))
        
        self.exit_button = ttk.Button(
            self.bottom_frame, 
            text="Exit", 
            command=self.root.destroy,
            style='Action.TButton',
            padding=(20, 10)
        )
        self.exit_button.pack(side=tk.RIGHT, padx=10)
        
        self.run_button = ttk.Button(
            self.bottom_frame, 
            text="Run Chess Game", 
            command=self._run_chess_game,
            style='Action.TButton',
            padding=(20, 10)
        )
        self.run_button.pack(side=tk.RIGHT, padx=10)
        
        # Initialize with available configurations
        self._refresh_config_list()
        self._refresh_ruleset_list()
        
    def _setup_load_tab(self):
        """Setup the Load Configuration tab"""
        # Frame for existing configurations
        saved_frame = ttk.LabelFrame(self.load_tab, text="Saved Configurations", padding=10)
        saved_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Dropdown for selecting saved configurations
        ttk.Label(saved_frame, text="Select a saved configuration:").pack(anchor=tk.W, pady=5)
        
        self.config_var = tk.StringVar()
        self.config_dropdown = ttk.Combobox(saved_frame, textvariable=self.config_var, state="readonly", width=50)
        self.config_dropdown.pack(fill=tk.X, pady=5)
        self.config_dropdown.bind("<<ComboboxSelected>>", self._on_config_selected)
        
        # Button frame
        button_frame = ttk.Frame(saved_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Buttons for load actions
        load_button = ttk.Button(button_frame, text="Load Selected", command=self._load_selected_config)
        load_button.pack(side=tk.LEFT, padx=5)
        
        refresh_button = ttk.Button(button_frame, text="Refresh List", command=self._refresh_config_list)
        refresh_button.pack(side=tk.LEFT, padx=5)
        
        delete_button = ttk.Button(button_frame, text="Delete Selected", command=self._delete_selected_config)
        delete_button.pack(side=tk.LEFT, padx=5)
        
        # Configuration preview
        preview_frame = ttk.LabelFrame(self.load_tab, text="Configuration Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD, width=80, height=15)
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        self.preview_text.configure(bg=self.colors["text_bg"], fg=self.colors["text_fg"], font=("Consolas", 10))
        self.preview_text.config(state=tk.DISABLED)
        
    def _setup_json_tab(self):
        """Setup the JSON Editor tab"""
        # Frame for configuration name
        name_frame = ttk.Frame(self.json_tab)
        name_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(name_frame, text="Configuration Name:").pack(side=tk.LEFT, padx=5)
        self.config_name_var = tk.StringVar()
        self.config_name_entry = ttk.Entry(name_frame, textvariable=self.config_name_var, width=40)
        self.config_name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # JSON editor
        editor_frame = ttk.LabelFrame(self.json_tab, text="JSON Configuration Editor", padding=10)
        editor_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.json_editor = scrolledtext.ScrolledText(editor_frame, wrap=tk.WORD)
        self.json_editor.pack(fill=tk.BOTH, expand=True)
        self.json_editor.configure(bg=self.colors["text_bg"], fg=self.colors["text_fg"], font=("Consolas", 10))
        
        # Default text
        self.json_editor.insert(tk.END, json.dumps(self.current_config, indent=4))
        
        # Button frame
        button_frame = ttk.Frame(self.json_tab)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Buttons for JSON actions
        validate_button = ttk.Button(button_frame, text="Validate JSON", command=self._validate_json)
        validate_button.pack(side=tk.LEFT, padx=5)
        
        save_button = ttk.Button(button_frame, text="Save Configuration", command=self._save_json_config)
        save_button.pack(side=tk.LEFT, padx=5)
        
        load_to_form_button = ttk.Button(button_frame, text="Load to Form", command=self._load_json_to_form)
        load_to_form_button.pack(side=tk.LEFT, padx=5)
        
    def _setup_form_tab(self):
        """Setup the Guided Configuration tab"""
        # Create a canvas with scrollbar for the form
        self.canvas = tk.Canvas(self.form_tab, bg=self.colors["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.form_tab, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # General Settings
        general_frame = ttk.LabelFrame(self.scrollable_frame, text="General Settings", padding=10)
        general_frame.pack(fill=tk.X, padx=5, pady=5, ipadx=5, ipady=5)
        
        engine_config = self.current_config["engine_config"]

        # Starting position
        ttk.Label(general_frame, text="Starting Position:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.start_pos_var = tk.StringVar(value=engine_config["starting_position"])
        start_pos_entry = ttk.Entry(general_frame, textvariable=self.start_pos_var, width=30)
        start_pos_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(general_frame, text="Default or FEN string").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        # White player
        ttk.Label(general_frame, text="White Player:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.white_player_var = tk.StringVar(value=engine_config["white_player"])
        white_player_combo = ttk.Combobox(general_frame, textvariable=self.white_player_var, values=self.engine_options, state="readonly", width=28)
        white_player_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Black player
        ttk.Label(general_frame, text="Black Player:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.black_player_var = tk.StringVar(value=engine_config["black_player"])
        black_player_combo = ttk.Combobox(general_frame, textvariable=self.black_player_var, values=self.engine_options, state="readonly", width=28)
        black_player_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Game count
        ttk.Label(general_frame, text="Game Count:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.game_count_var = tk.IntVar(value=engine_config["game_count"])
        game_count_spin = ttk.Spinbox(general_frame, from_=1, to=100, textvariable=self.game_count_var, width=5)
        game_count_spin.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Engine Configuration
        engine_frame = ttk.LabelFrame(self.scrollable_frame, text="Engine Configuration", padding=10)
        engine_frame.pack(fill=tk.X, padx=5, pady=5, ipadx=5, ipady=5)
        
        engine_config = self.current_config["engine_config"]
        
        # Engine name
        ttk.Label(engine_frame, text="Engine Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.engine_name_var = tk.StringVar(value=engine_config["name"])
        engine_name_entry = ttk.Entry(engine_frame, textvariable=self.engine_name_var, width=30)
        engine_name_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Engine version
        ttk.Label(engine_frame, text="Engine Version:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.engine_version_var = tk.StringVar(value=engine_config["version"])
        engine_version_entry = ttk.Entry(engine_frame, textvariable=self.engine_version_var, width=30)
        engine_version_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Engine color
        ttk.Label(engine_frame, text="Engine Color:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.engine_color_var = tk.StringVar(value=engine_config["color"])
        engine_color_combo = ttk.Combobox(engine_frame, textvariable=self.engine_color_var, values=["white", "black"], state="readonly", width=28)
        engine_color_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Ruleset
        ttk.Label(engine_frame, text="Ruleset:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.ruleset_var = tk.StringVar(value=engine_config["ruleset"])
        ruleset_combo = ttk.Combobox(engine_frame, textvariable=self.ruleset_var, values=self.rulesets, state="readonly", width=28)
        ruleset_combo.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Add buttons to manage rulesets
        ruleset_button_frame = ttk.Frame(engine_frame)
        ruleset_button_frame.grid(row=3, column=2, sticky=tk.W, padx=5, pady=5)
        
        edit_ruleset_button = ttk.Button(ruleset_button_frame, text="Edit", 
                                         command=lambda: self._edit_selected_engine_ruleset())
        edit_ruleset_button.pack(side=tk.LEFT, padx=2)
        
        new_ruleset_button = ttk.Button(ruleset_button_frame, text="New", 
                                        command=lambda: self._new_ruleset_from_form())
        new_ruleset_button.pack(side=tk.LEFT, padx=2)
        
        # Search algorithm
        ttk.Label(engine_frame, text="Search Algorithm:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.search_algorithm_var = tk.StringVar(value=engine_config["search_algorithm"])
        search_algorithm_combo = ttk.Combobox(engine_frame, textvariable=self.search_algorithm_var, 
                                              values=["lookahead", "minimax", "alphabeta", "quiescence", "iterative_deepening"], 
                                              state="readonly", width=28)
        search_algorithm_combo.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Search depth
        ttk.Label(engine_frame, text="Search Depth:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.depth_var = tk.IntVar(value=engine_config["depth"])
        depth_spin = ttk.Spinbox(engine_frame, from_=1, to=20, textvariable=self.depth_var, width=5)
        depth_spin.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Max depth
        ttk.Label(engine_frame, text="Max Depth:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_depth_var = tk.IntVar(value=engine_config["max_depth"])
        max_depth_spin = ttk.Spinbox(engine_frame, from_=1, to=30, textvariable=self.max_depth_var, width=5)
        max_depth_spin.grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Monitoring enabled
        ttk.Label(engine_frame, text="Monitoring Enabled:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=5)
        self.monitoring_var = tk.BooleanVar(value=engine_config["monitoring_enabled"])
        monitoring_check = ttk.Checkbutton(engine_frame, variable=self.monitoring_var)
        monitoring_check.grid(row=7, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Verbose output
        ttk.Label(engine_frame, text="Verbose Output:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
        self.verbose_var = tk.BooleanVar(value=engine_config["verbose_output"])
        verbose_check = ttk.Checkbutton(engine_frame, variable=self.verbose_var)
        verbose_check.grid(row=8, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Logger
        ttk.Label(engine_frame, text="Logger:").grid(row=9, column=0, sticky=tk.W, padx=5, pady=5)
        self.logger_var = tk.StringVar(value=engine_config["logger"])
        logger_entry = ttk.Entry(engine_frame, textvariable=self.logger_var, width=30)
        logger_entry.grid(row=9, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Stockfish Configuration
        stockfish_frame = ttk.LabelFrame(self.scrollable_frame, text="Stockfish Configuration", padding=10)
        stockfish_frame.pack(fill=tk.X, padx=5, pady=5, ipadx=5, ipady=5)
        
        stockfish_config = self.current_config["stockfish_config"]
        
        # Stockfish path
        ttk.Label(stockfish_frame, text="Stockfish Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.stockfish_path_var = tk.StringVar(value=stockfish_config["stockfish_path"])
        stockfish_path_entry = ttk.Entry(stockfish_frame, textvariable=self.stockfish_path_var, width=50)
        stockfish_path_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        browse_button = ttk.Button(stockfish_frame, text="Browse", command=self._browse_stockfish)
        browse_button.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        # ELO rating
        ttk.Label(stockfish_frame, text="ELO Rating:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.elo_var = tk.IntVar(value=stockfish_config["elo_rating"])
        elo_spin = ttk.Spinbox(stockfish_frame, from_=100, to=3000, textvariable=self.elo_var, width=5)
        elo_spin.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Skill level
        ttk.Label(stockfish_frame, text="Skill Level:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.skill_var = tk.IntVar(value=stockfish_config["skill_level"])
        skill_spin = ttk.Spinbox(stockfish_frame, from_=0, to=20, textvariable=self.skill_var, width=5)
        skill_spin.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Debug mode
        ttk.Label(stockfish_frame, text="Debug Mode:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.debug_var = tk.BooleanVar(value=stockfish_config["debug_mode"])
        debug_check = ttk.Checkbutton(stockfish_frame, variable=self.debug_var)
        debug_check.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Depth
        ttk.Label(stockfish_frame, text="Depth:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.sf_depth_var = tk.IntVar(value=stockfish_config["depth"])
        sf_depth_spin = ttk.Spinbox(stockfish_frame, from_=1, to=20, textvariable=self.sf_depth_var, width=5)
        sf_depth_spin.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Max depth
        ttk.Label(stockfish_frame, text="Max Depth:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.sf_max_depth_var = tk.IntVar(value=stockfish_config["max_depth"])
        sf_max_depth_spin = ttk.Spinbox(stockfish_frame, from_=1, to=30, textvariable=self.sf_max_depth_var, width=5)
        sf_max_depth_spin.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Move time
        ttk.Label(stockfish_frame, text="Move Time (ms):").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.movetime_var = tk.IntVar(value=stockfish_config["movetime"])
        movetime_spin = ttk.Spinbox(stockfish_frame, from_=100, to=10000, textvariable=self.movetime_var, increment=100, width=5)
        movetime_spin.grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Buttons for form actions
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.form_name_var = tk.StringVar()
        ttk.Label(button_frame, text="Configuration Name:").pack(side=tk.LEFT, padx=5)
        form_name_entry = ttk.Entry(button_frame, textvariable=self.form_name_var, width=30)
        form_name_entry.pack(side=tk.LEFT, padx=5)
        
        save_form_button = ttk.Button(button_frame, text="Save Form Configuration", command=self._save_form_config)
        save_form_button.pack(side=tk.LEFT, padx=5)
        
        load_to_json_button = ttk.Button(button_frame, text="Load to JSON Editor", command=self._load_form_to_json)
        load_to_json_button.pack(side=tk.LEFT, padx=5)
        
    def _setup_ruleset_tab(self):
        """Setup the Ruleset Manager tab"""
        # Frame for existing rulesets
        saved_frame = ttk.LabelFrame(self.ruleset_tab, text="Saved Rulesets", padding=10)
        saved_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Dropdown for selecting saved rulesets
        ttk.Label(saved_frame, text="Select a saved ruleset:").pack(anchor=tk.W, pady=5)
        
        self.ruleset_dropdown_var = tk.StringVar()
        self.ruleset_combo = ttk.Combobox(saved_frame, textvariable=self.ruleset_dropdown_var, state="readonly", width=50)
        self.ruleset_combo.pack(fill=tk.X, pady=5)
        self.ruleset_combo.bind("<<ComboboxSelected>>", self._on_ruleset_selected)
        
        # Button frame
        button_frame = ttk.Frame(saved_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Buttons for ruleset actions
        load_button = ttk.Button(button_frame, text="Load Selected", command=self._load_selected_ruleset)
        load_button.pack(side=tk.LEFT, padx=5)
        
        refresh_button = ttk.Button(button_frame, text="Refresh List", command=self._refresh_ruleset_list)
        refresh_button.pack(side=tk.LEFT, padx=5)
        
        new_button = ttk.Button(button_frame, text="New Ruleset", command=self._new_ruleset)
        new_button.pack(side=tk.LEFT, padx=5)
        
        # Ruleset preview
        preview_frame = ttk.LabelFrame(self.ruleset_tab, text="Ruleset Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.ruleset_preview = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD, width=80, height=15)
        self.ruleset_preview.pack(fill=tk.BOTH, expand=True)
        self.ruleset_preview.configure(bg=self.colors["text_bg"], fg=self.colors["text_fg"], font=("Consolas", 10))
        self.ruleset_preview.config(state=tk.DISABLED)
        
        # Initialize with available rulesets
        self._refresh_ruleset_list()
    
    def _setup_ruleset_editor_tab(self):
        """Setup the Ruleset Editor tab"""
        # Frame for ruleset name
        name_frame = ttk.Frame(self.ruleset_editor_tab)
        name_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(name_frame, text="Ruleset Name:").pack(side=tk.LEFT, padx=5)
        self.ruleset_name_var = tk.StringVar()
        self.ruleset_name_entry = ttk.Entry(name_frame, textvariable=self.ruleset_name_var, width=40)
        self.ruleset_name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Buttons for ruleset actions are at the bottom
        button_frame = ttk.Frame(self.ruleset_editor_tab)
        button_frame.pack(fill=tk.X, pady=10, padx=5, side=tk.BOTTOM)
        
        save_button = ttk.Button(button_frame, text="Save Ruleset", command=self._save_ruleset_editor)
        save_button.pack(side=tk.LEFT, padx=5)
        
        load_button = ttk.Button(button_frame, text="Load Ruleset", command=self._load_ruleset_editor)
        load_button.pack(side=tk.LEFT, padx=5)
        
        reset_button = ttk.Button(button_frame, text="Reset to Default", command=self._reset_ruleset_editor)
        reset_button.pack(side=tk.LEFT, padx=5)
        
        # A frame to contain the canvas and scrollbar, this will fill the remaining space
        canvas_container = ttk.Frame(self.ruleset_editor_tab)
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create a canvas with scrollbar for the form
        self.ruleset_canvas = tk.Canvas(canvas_container, bg=self.colors["bg"], highlightthickness=0)
        self.ruleset_scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.ruleset_canvas.yview)
        self.ruleset_scrollable_frame = ttk.Frame(self.ruleset_canvas)
        
        self.ruleset_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.ruleset_canvas.configure(scrollregion=self.ruleset_canvas.bbox("all"))
        )
        
        self.ruleset_canvas.create_window((0, 0), window=self.ruleset_scrollable_frame, anchor="nw")
        self.ruleset_canvas.configure(yscrollcommand=self.ruleset_scrollbar.set)
        
        self.ruleset_scrollbar.pack(side="right", fill="y")
        self.ruleset_canvas.pack(side="left", fill="both", expand=True)
        
        # Group parameters by category for better organization
        self._create_ruleset_editor_fields()
    
    def _create_ruleset_editor_fields(self):
        """Create the form fields for ruleset parameters"""
        # Define parameter categories for better organization
        categories = {
            "Piece Values": [
                "material_weight",
            ],
            "Pawn Structure": [
                "backward_pawn_penalty", "doubled_pawn_penalty", "isolated_pawn_penalty",
                "passed_pawn_bonus", "pawn_advancement_bonus", "pawn_promotion_bonus",
                "pawn_structure_bonus", "en_passant_bonus"
            ],
            "Piece Activity": [
                "bishop_activity_bonus", "bishop_vision_bonus", "knight_activity_bonus",
                "knight_pair_bonus", "knight_vision_penalty", "piece_activity_bonus",
                "piece_coordination_bonus", "piece_development_bonus", "piece_mobility_bonus",
                "undeveloped_penalty"
            ],
            "Board Control": [
                "center_control_bonus", "file_control_bonus", "open_file_bonus",
                "coordinated_rooks_bonus", "rook_development_penalty", "rook_position_bonus",
                "stacked_rooks_bonus", "tempo_bonus"
            ],
            "King Safety": [
                "castling_bonus", "castling_protection_bonus", "castling_protection_penalty",
                "exposed_king_penalty", "king_safety_bonus", "king_safety_penalty",
                "king_threat_penalty"
            ],
            "Tactical Elements": [
                "capture_bonus", "capture_move_bonus", "check_bonus", "check_move_bonus",
                "checkmate_bonus", "checkmate_move_bonus", "counter_move_bonus",
                "draw_penalty", "hanging_piece_bonus", "hash_move_bonus",
                "history_move_bonus", "in_check_penalty", "killer_move_bonus",
                "promotion_move_bonus", "queen_capture_bonus", "repetition_penalty",
                "stalemate_penalty", "trapped_piece_penalty", "undefended_piece_penalty"
            ]
        }
        
        # Parameter descriptions
        descriptions = {
            "material_weight": "Weight of material value in the evaluation (0-10)",
            "backward_pawn_penalty": "Penalty for pawns that are behind friendly pawns and difficult to advance",
            "doubled_pawn_penalty": "Penalty for having two pawns on the same file",
            "isolated_pawn_penalty": "Penalty for pawns with no friendly pawns on adjacent files",
            "passed_pawn_bonus": "Bonus for pawns with no opposing pawns blocking promotion path",
            "pawn_advancement_bonus": "Bonus for pawns that have advanced far into enemy territory",
            "pawn_promotion_bonus": "Bonus for pawns close to promotion",
            "pawn_structure_bonus": "Bonus for strong pawn structure (chains, defended pawns)",
            "en_passant_bonus": "Bonus for en passant capture opportunities",
            "bishop_activity_bonus": "Bonus for active bishops controlling many squares",
            "bishop_vision_bonus": "Bonus for bishops with clear diagonals",
            "knight_activity_bonus": "Bonus for knights in central/active positions",
            "knight_pair_bonus": "Bonus for having both knights",
            "knight_vision_penalty": "Penalty for knights with restricted movement",
            "piece_activity_bonus": "General bonus for active pieces",
            "piece_coordination_bonus": "Bonus for pieces supporting each other",
            "piece_development_bonus": "Bonus for developed pieces in the opening",
            "piece_mobility_bonus": "Bonus for pieces with many available moves",
            "undeveloped_penalty": "Penalty for pieces not yet developed",
            "center_control_bonus": "Bonus for controlling central squares",
            "file_control_bonus": "Bonus for controlling files",
            "open_file_bonus": "Bonus for rooks on open files",
            "coordinated_rooks_bonus": "Bonus for connected rooks",
            "rook_development_penalty": "Penalty for undeveloped rooks",
            "rook_position_bonus": "Bonus for rooks in strong positions",
            "stacked_rooks_bonus": "Bonus for rooks on the same file",
            "tempo_bonus": "Bonus for the side to move",
            "castling_bonus": "Bonus for having castled",
            "castling_protection_bonus": "Bonus for protection after castling",
            "castling_protection_penalty": "Penalty for exposed king after castling",
            "exposed_king_penalty": "Penalty for king exposed to attack",
            "king_safety_bonus": "Bonus for having a protected king",
            "king_safety_penalty": "Penalty for unsafe king position",
            "king_threat_penalty": "Penalty for direct threats to the king",
            "capture_bonus": "Bonus for capturing pieces",
            "capture_move_bonus": "Bonus for moves that capture pieces",
            "check_bonus": "Bonus for giving check",
            "check_move_bonus": "Bonus for moves that give check",
            "checkmate_bonus": "Bonus for checkmate position",
            "checkmate_move_bonus": "Bonus for moves that give checkmate",
            "counter_move_bonus": "Bonus for counter moves",
            "draw_penalty": "Penalty for drawing positions",
            "hanging_piece_bonus": "Bonus for attacking hanging pieces",
            "hash_move_bonus": "Bonus for hash table moves",
            "history_move_bonus": "Bonus for historically good moves",
            "in_check_penalty": "Penalty for being in check",
            "killer_move_bonus": "Bonus for killer moves in search",
            "promotion_move_bonus": "Bonus for moves that promote pawns",
            "queen_capture_bonus": "Bonus for capturing the opponent's queen",
            "repetition_penalty": "Penalty for move repetition",
            "stalemate_penalty": "Penalty for stalemate positions",
            "trapped_piece_penalty": "Penalty for pieces with limited mobility",
            "undefended_piece_penalty": "Penalty for pieces not defended"
        }
        
        # Get default values
        default_values = self.current_ruleset
        
        # Create variables for all parameters
        self.ruleset_vars = {}

        # Clear any existing widgets in the frame before creating new ones
        for widget in self.ruleset_scrollable_frame.winfo_children():
            widget.destroy()
        
        # Create forms for each category
        for category, params in categories.items():
            # Create category frame
            category_frame = ttk.LabelFrame(self.ruleset_scrollable_frame, text=category, padding=10)
            category_frame.pack(fill=tk.X, padx=5, pady=5, ipadx=5, ipady=5)
            category_frame.columnconfigure(1, weight=1) # Allow widget column to expand
            
            # Add parameters
            for i, param in enumerate(params):
                current_row = i * 2
                ttk.Label(category_frame, text=param.replace('_', ' ').title() + ":").grid(row=current_row, column=0, sticky=tk.W, padx=5, pady=3)
                
                # Create variable based on parameter type
                value = default_values.get(param, 0)
                if isinstance(value, bool):
                    var = tk.BooleanVar(value=value)
                    widget = ttk.Checkbutton(category_frame, variable=var)
                elif isinstance(value, int):
                    var = tk.IntVar(value=value)
                    widget = ttk.Spinbox(category_frame, from_=-10000000000, to=10000000000, textvariable=var, width=20)
                else:  # float
                    var = tk.DoubleVar(value=value)
                    widget = ttk.Spinbox(category_frame, from_=-10000000000.0, to=10000000000.0, textvariable=var, width=20, increment=0.01)
                
                self.ruleset_vars[param] = var
                widget.grid(row=current_row, column=1, sticky=tk.W, padx=5, pady=3)
                
                # Add description label
                desc = descriptions.get(param, "")
                desc_label = ttk.Label(category_frame, text=desc, wraplength=500, justify=tk.LEFT)
                desc_label.grid(row=current_row + 1, column=0, columnspan=2, sticky=tk.W, padx=15, pady=(0, 5))
    
    def _show_status(self, message, is_error=False):
        """Display a status message or error in the window title"""
        if is_error:
            messagebox.showerror("Error", message)
        else:
            # Update the window title with the status message
            original_title = "V7P3R Chess Engine Configuration"
            self.root.title(f"{original_title} - {message}")
            # Schedule title reset after 3 seconds
            self.root.after(3000, lambda: self.root.title(original_title))
            
    def _refresh_config_list(self):
        """Refresh the list of saved configurations"""
        configs = list_saved_configs()
        self.config_dropdown['values'] = configs
        if configs:
            self.config_var.set(configs[0])
            self._on_config_selected(None)
        else:
            self.config_var.set("")
            self.preview_text.config(state=tk.NORMAL)
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, "No saved configurations found.")
            self.preview_text.config(state=tk.DISABLED)
    
    def _refresh_ruleset_list(self):
        """Refresh the list of available rulesets"""
        print("Refreshing ruleset list")
        self.ruleset_data = load_ruleset_data()
        ruleset_names = list(self.ruleset_data.keys())
        self.ruleset_combo['values'] = ruleset_names
        
        if ruleset_names:
            self.ruleset_dropdown_var.set(ruleset_names[0])
            print(f"Setting initial ruleset to: {ruleset_names[0]}")
            try:
                self._on_ruleset_selected(None)
                print("Ruleset selection handled successfully")
            except Exception as e:
                print(f"Error in _on_ruleset_selected: {e}")
        else:
            self.ruleset_dropdown_var.set("")
            print("No rulesets found")
            self.ruleset_preview.config(state=tk.NORMAL)
            self.ruleset_preview.delete(1.0, tk.END)
            self.ruleset_preview.insert(tk.END, "No rulesets found.")
            self.ruleset_preview.config(state=tk.DISABLED)
    
    def _on_config_selected(self, event):
        """Handle configuration selection from dropdown"""
        selected = self.config_var.get()
        if selected:
            config_data, error = load_config(selected)
            if config_data:
                self.preview_text.config(state=tk.NORMAL)
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(tk.END, json.dumps(config_data, indent=4))
                self.preview_text.config(state=tk.DISABLED)
            else:
                messagebox.showerror("Error", f"Failed to load configuration: {error}")
    
    def _on_ruleset_selected(self, event):
        """Handle ruleset selection from dropdown"""
        print("Ruleset selection event triggered")
        selected = self.ruleset_dropdown_var.get()
        print(f"Selected ruleset: {selected}")
        
        if selected and selected in self.ruleset_data:
            ruleset_data = self.ruleset_data[selected]
            self.ruleset_preview.config(state=tk.NORMAL)
            self.ruleset_preview.delete(1.0, tk.END)
            
            # Format the ruleset for display
            yaml_text = yaml.dump({selected: ruleset_data}, default_flow_style=False, sort_keys=False)
            self.ruleset_preview.insert(tk.END, yaml_text)
            self.ruleset_preview.config(state=tk.DISABLED)
            
            # Status message removed - using messagebox for important notifications only
    
    def _load_selected_config(self):
        """Load the selected configuration"""
        selected = self.config_var.get()
        if not selected:
            messagebox.showwarning("Warning", "No configuration selected.")
            return
        
        config_data, error = load_config(selected)
        if config_data:
            self.current_config = config_data
            
            # Update JSON tab
            self.config_name_var.set(selected.replace('.json', ''))
            self.json_editor.delete(1.0, tk.END)
            self.json_editor.insert(tk.END, json.dumps(config_data, indent=4))
            
            # Update form tab with values from config
            self._update_form_from_config(config_data)
            
            # Status message removed - using messagebox for important notifications only
            
            # Switch to JSON tab
            self.notebook.select(1)
        else:
            messagebox.showerror("Error", f"Failed to load configuration: {error}")
    
    def _load_selected_ruleset(self):
        """Load the selected ruleset into the editor"""
        selected = self.ruleset_dropdown_var.get()
        if not selected:
            messagebox.showwarning("Warning", "No ruleset selected.")
            return
        
        if selected in self.ruleset_data:
            self.current_ruleset = self.ruleset_data[selected]
            self.ruleset_name_var.set(selected)
            
            # Update editor fields
            for param, var in self.ruleset_vars.items():
                if param in self.current_ruleset:
                    var.set(self.current_ruleset[param])
            
            # Status message removed - using messagebox for important notifications only
            self.notebook.select(4)  # Switch to ruleset editor tab
        else:
            messagebox.showerror("Error", f"Ruleset '{selected}' not found.")
    
    def _new_ruleset(self):
        """Create a new ruleset with default values"""
        self.current_ruleset = get_default_ruleset_values()
        self.ruleset_name_var.set("")
        
        # Update editor fields
        for param, var in self.ruleset_vars.items():
            if param in self.current_ruleset:
                var.set(self.current_ruleset[param])
        
        # Status message removed - using messagebox for important notifications only
        self.notebook.select(4)  # Switch to ruleset editor tab
    
    def _save_ruleset_editor(self):
        """Save the current ruleset from editor"""
        ruleset_name = self.ruleset_name_var.get()
        
        if not ruleset_name:
            # Generate timestamp name if none provided
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            ruleset_name = f"custom_evaluation_{timestamp}"
            self.ruleset_name_var.set(ruleset_name)
        
        # Collect values from form
        ruleset_data = {}
        for param, var in self.ruleset_vars.items():
            ruleset_data[param] = var.get()
        
        # Save the ruleset
        success, result = save_ruleset(ruleset_name, ruleset_data, self.ruleset_data)
        
        if success:
            self.current_ruleset = ruleset_data
            messagebox.showinfo("Success", f"Ruleset saved as '{result}'")
            # Status message removed - using messagebox for important notifications only
            self._refresh_ruleset_list()
            self.ruleset_var.set(result)
        else:
            messagebox.showerror("Error", f"Failed to save ruleset: {result}")
    
    def _load_ruleset_editor(self):
        """Load a ruleset into the editor (redirects to the ruleset selection tab)"""
        self.notebook.select(3)  # Switch to ruleset manager tab
        # Status message removed - using messagebox for important notifications only
    
    def _reset_ruleset_editor(self):
        """Reset the ruleset editor to default values"""
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset to default values?"):
            self.current_ruleset = get_default_ruleset_values()
            
            # Update editor fields
            for param, var in self.ruleset_vars.items():
                if param in self.current_ruleset:
                    var.set(self.current_ruleset[param])
            
            # Status message removed - using messagebox for important notifications only
            

    def _browse_stockfish(self):
        """Browse for Stockfish executable"""
        stockfish_path = filedialog.askopenfilename(
            title="Select Stockfish Executable",
            filetypes=[("Executables", "*.exe"), ("All Files", "*.*")]
        )
        if stockfish_path:
            self.stockfish_path_var.set(stockfish_path)
    
    def _run_chess_game(self):
        """Run the chess game with the current configuration"""
        # Determine which tab is active and get the appropriate configuration
        current_tab = self.notebook.index(self.notebook.select())
        config_data = None
        
        try:
            if current_tab == 0:  # Load Configuration tab
                selected = self.config_var.get()
                if not selected:
                    messagebox.showwarning("Warning", "No configuration selected.")
                    return
                
                config_data, error = load_config(selected)
                if error:
                    messagebox.showerror("Error", f"Failed to load configuration: {error}")
                    return
            
            elif current_tab == 1:  # JSON Editor tab
                if not self._validate_json():
                    return
                
                json_text = self.json_editor.get(1.0, tk.END)
                try:
                    config_data = json.loads(json_text)
                except json.JSONDecodeError as e:
                    messagebox.showerror("JSON Error", f"Invalid JSON: {str(e)}")
                    return
            
            elif current_tab == 2:  # Guided Configuration tab
                # Construct config from form fields
                config_data = {
                    "starting_position": self.start_pos_var.get(),
                    "white_player": self.white_player_var.get(),
                    "black_player": self.black_player_var.get(),
                    "game_count": int(self.game_count_var.get()),
                    "engine_config": {
                        "name": self.engine_name_var.get(),
                        "version": self.engine_version_var.get(),
                        "color": self.engine_color_var.get(),
                        "ruleset": self.ruleset_var.get(),
                        "search_algorithm": self.search_algorithm_var.get(),
                        "depth": int(self.depth_var.get()),
                        "max_depth": int(self.max_depth_var.get()),
                        "monitoring_enabled": bool(self.monitoring_var.get()),
                        "verbose_output": bool(self.verbose_var.get()),
                        "logger": self.logger_var.get(),
                        "starting_position": self.start_pos_var.get(),
                        "white_player": self.white_player_var.get(),
                        "black_player": self.black_player_var.get(),
                    },
                    "stockfish_config": {
                        "stockfish_path": self.stockfish_path_var.get(),
                        "elo_rating": int(self.elo_var.get()),
                        "skill_level": int(self.skill_var.get()),
                        "debug_mode": bool(self.debug_var.get()),
                        "depth": int(self.sf_depth_var.get()),
                        "max_depth": int(self.sf_max_depth_var.get()),
                        "movetime": int(self.movetime_var.get()),
                    },
                }
            
            elif current_tab == 3 or current_tab == 4:  # Ruleset Manager or Ruleset Editor tab
                # For ruleset tabs, prompt user to select a configuration first
                messagebox.showinfo("Select Configuration", "Please select or create a configuration first.")
                self.notebook.select(0)  # Switch to Load Configuration tab
                return

            # Make sure config_data is not None
            if config_data is None:
                messagebox.showerror("Error", "Failed to get configuration data")
                return

            # Check if the selected ruleset exists before running
            ruleset_name = config_data.get("engine_config", {}).get("ruleset", "default_evaluation")
            ruleset_data = load_ruleset_data()
            
            if ruleset_name not in ruleset_data:
                result = messagebox.askyesno(
                    "Ruleset Not Found", 
                    f"The selected ruleset '{ruleset_name}' was not found. Do you want to use 'default_evaluation' instead?"
                )
                if result:
                    config_data["engine_config"]["ruleset"] = "default_evaluation"
                else:
                    return
            
            # Save configuration before running
            if messagebox.askyesno("Save Configuration", "Do you want to save this configuration before running?"):
                config_name = ""
                if current_tab == 0:
                    config_name = selected
                elif current_tab == 1:
                    config_name = self.config_name_var.get()
                elif current_tab == 2:
                    config_name = self.form_name_var.get()
                
                success, result = save_config(config_name, config_data)
                if success:
                    self._show_status(f"Configuration saved as '{result}' before running")
                    self._refresh_config_list()
                else:
                    messagebox.showerror("Error", f"Failed to save configuration: {result}")
                    # Continue anyway
            
            # Run the game
            try:
                messagebox.showinfo("Starting Game", "Starting chess game...")
                self.root.update_idletasks()  # Force UI update
                
                # Import ChessGame here to avoid circular imports
                try:
                    # Try local import first
                    from v7p3r_engine.v7p3r_play import ChessGame
                except ImportError:
                    # Fallback to package import
                    from v7p3r_engine.v7p3r_play import ChessGame
                
                game = ChessGame(config_data)
                game.run()
                
                # Close metrics store if it exists
                if hasattr(game, 'metrics_store') and game.metrics_store:
                    game.metrics_store.close()
                
                messagebox.showinfo("Game Complete", "Chess game completed successfully.")
            except ImportError as ie:
                messagebox.showerror("Import Error", f"Could not import ChessGame: {ie}\n\nMake sure play_v7p3r.py is in the correct location.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to run chess game: {e}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error setting up game: {e}")
    
    def _validate_json(self):
        """Validate the JSON in the editor"""
        json_text = self.json_editor.get(1.0, tk.END)
        try:
            config_data = json.loads(json_text)
            valid, error = validate_config(config_data)
            if valid:
                messagebox.showinfo("Validation", "Configuration is valid.")
                return True
            else:
                messagebox.showerror("Validation Error", error)
                return False
        except json.JSONDecodeError as e:
            messagebox.showerror("JSON Error", f"Invalid JSON: {str(e)}")
            return False
    
    def _save_json_config(self):
        """Save the JSON configuration"""
        if not self._validate_json():
            return
        
        json_text = self.json_editor.get(1.0, tk.END)
        config_data = json.loads(json_text)
        config_name = self.config_name_var.get()
        
        success, result = save_config(config_name, config_data)
        if success:
            self.current_config = config_data
            messagebox.showinfo("Success", f"Configuration saved as '{result}'")
            # Status message removed - using messagebox for important notifications only
            self._refresh_config_list()
            self.config_var.set(result.replace('.json', ''))
        else:
            messagebox.showerror("Error", f"Failed to save configuration: {result}")
            # Status message removed - using messagebox for important notifications only
    
    def _load_json_to_form(self):
        """Load JSON data to the form fields"""
        if not self._validate_json():
            return
        
        json_text = self.json_editor.get(1.0, tk.END)
        try:
            config_data = json.loads(json_text)
            self._update_form_from_config(config_data)
            self.form_name_var.set(self.config_name_var.get())
            self.notebook.select(2)  # Switch to form tab
            # Status message removed - using messagebox for important notifications only
            messagebox.showinfo("Form Loaded", "JSON loaded to form fields")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load JSON to form: {e}")
            # Status message removed - using messagebox for important notifications only
            messagebox.showerror("Load Error", f"Form load error: {e}")
    
    def _update_form_from_config(self, config_data):
        """Update form fields from configuration data"""
        try:
            # General settings
            self.start_pos_var.set(config_data.get("starting_position", "default"))
            self.white_player_var.set(config_data.get("white_player", "v7p3r"))
            self.black_player_var.set(config_data.get("black_player", "stockfish"))
            self.game_count_var.set(config_data.get("game_count", 1))
            
            # Engine config
            engine_config = config_data.get("engine_config", {})
            self.engine_name_var.set(engine_config.get("name", "v7p3r"))
            self.engine_version_var.set(engine_config.get("version", "1.0.0"))
            self.engine_color_var.set(engine_config.get("color", "white"))
            self.ruleset_var.set(engine_config.get("ruleset", "default_evaluation"))
            self.search_algorithm_var.set(engine_config.get("search_algorithm", "lookahead"))
            self.depth_var.set(engine_config.get("depth", 5))
            self.max_depth_var.set(engine_config.get("max_depth", 8))
            self.monitoring_var.set(engine_config.get("monitoring_enabled", True))
            self.verbose_var.set(engine_config.get("verbose_output", True))
            self.logger_var.set(engine_config.get("logger", "v7p3r_engine_logger"))
            
            # Stockfish config
            stockfish_config = config_data.get("stockfish_config", {})
            self.stockfish_path_var.set(stockfish_config.get("stockfish_path", "engine_utilities/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe"))
            self.elo_var.set(stockfish_config.get("elo_rating", 100))
            self.skill_var.set(stockfish_config.get("skill_level", 1))
            self.debug_var.set(stockfish_config.get("debug_mode", False))
            self.sf_depth_var.set(stockfish_config.get("depth", 2))
            self.sf_max_depth_var.set(stockfish_config.get("max_depth", 3))
            self.movetime_var.set(stockfish_config.get("movetime", 500))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update form fields: {e}")
    
    def _save_form_config(self):
        """Save configuration from form fields"""
        try:
            # Construct config from form fields
            config_data = {
                "starting_position": self.start_pos_var.get(),
                "white_player": self.white_player_var.get(),
                "black_player": self.black_player_var.get(),
                "game_count": self.game_count_var.get(),
                "engine_config": {
                    "name": self.engine_name_var.get(),
                    "version": self.engine_version_var.get(),
                    "color": self.engine_color_var.get(),
                    "ruleset": self.ruleset_var.get(),
                    "search_algorithm": self.search_algorithm_var.get(),
                    "depth": self.depth_var.get(),
                    "max_depth": self.max_depth_var.get(),
                    "monitoring_enabled": self.monitoring_var.get(),
                    "verbose_output": self.verbose_var.get(),
                    "logger": self.logger_var.get(),
                    "starting_position": self.start_pos_var.get(),
                    "white_player": self.white_player_var.get(),
                    "black_player": self.black_player_var.get(),
                },
                "stockfish_config": {
                    "stockfish_path": self.stockfish_path_var.get(),
                    "elo_rating": self.elo_var.get(),
                    "skill_level": self.skill_var.get(),
                    "debug_mode": self.debug_var.get(),
                    "depth": self.sf_depth_var.get(),
                    "max_depth": self.sf_max_depth_var.get(),
                    "movetime": self.movetime_var.get(),
                },
            }
            
            # Validate the configuration
            valid, error = validate_config(config_data)
            if not valid:
                messagebox.showerror("Validation Error", error)
                return
            
            # Save the configuration
            config_name = self.form_name_var.get()
            success, result = save_config(config_name, config_data)
            
            if success:
                self.current_config = config_data
                messagebox.showinfo("Success", f"Configuration saved as '{result}'")
                # Status message removed - using messagebox for important notifications only
                
                # Update the JSON editor with the new configuration
                self.config_name_var.set(self.form_name_var.get())
                self.json_editor.delete(1.0, tk.END)
                self.json_editor.insert(tk.END, json.dumps(config_data, indent=4))
                
                self._refresh_config_list()
                self.config_var.set(result.replace('.json', ''))
            else:
                messagebox.showerror("Error", f"Failed to save configuration: {result}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save form configuration: {e}")
    
    def _load_form_to_json(self):
        """Load form data to the JSON editor"""
        try:
            # Construct config from form fields
            config_data = {
                "starting_position": self.start_pos_var.get(),
                "white_player": self.white_player_var.get(),
                "black_player": self.black_player_var.get(),
                "game_count": self.game_count_var.get(),
                "engine_config": {
                    "name": self.engine_name_var.get(),
                    "version": self.engine_version_var.get(),
                    "color": self.engine_color_var.get(),
                    "ruleset": self.ruleset_var.get(),
                    "search_algorithm": self.search_algorithm_var.get(),
                    "depth": self.depth_var.get(),
                    "max_depth": self.max_depth_var.get(),
                    "monitoring_enabled": self.monitoring_var.get(),
                    "verbose_output": self.verbose_var.get(),
                    "logger": self.logger_var.get(),
                    "starting_position": self.start_pos_var.get(),
                    "white_player": self.white_player_var.get(),
                    "black_player": self.black_player_var.get(),
                },
                "stockfish_config": {
                    "stockfish_path": self.stockfish_path_var.get(),
                    "elo_rating": self.elo_var.get(),
                    "skill_level": self.skill_var.get(),
                    "debug_mode": self.debug_var.get(),
                    "depth": self.sf_depth_var.get(),
                    "max_depth": self.sf_max_depth_var.get(),
                    "movetime": self.movetime_var.get(),
                },
            }
            
            # Update the JSON editor
            self.config_name_var.set(self.form_name_var.get())
            self.json_editor.delete(1.0, tk.END)
            self.json_editor.insert(tk.END, json.dumps(config_data, indent=4))
            
            # Switch to JSON tab
            self.notebook.select(1)
            # Status message removed - using messagebox for important notifications only
            messagebox.showinfo("JSON Update", "Form data loaded to JSON editor")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load form data to JSON: {e}")
    
    def _delete_selected_config(self):
        """Delete the selected configuration"""
        selected = self.config_var.get()
        if not selected:
            messagebox.showwarning("Warning", "No configuration selected.")
            return
        
        confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{selected}'?")
        if confirm:
            try:
                filepath = os.path.join(CONFIG_DIR, selected + '.json')
                if os.path.exists(filepath):
                    os.remove(filepath)
                    self._refresh_config_list()
                    # Status message removed - using messagebox for important notifications only
                    messagebox.showinfo("Config Deleted", f"Deleted configuration: {selected}")
                else:
                    messagebox.showerror("Error", f"File not found: {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete configuration: {e}")

    def _new_ruleset_from_form(self):
        """Create a new ruleset using the current form values."""
        try:
            # Collect values from form fields
            ruleset_data = {}
            for param, var in self.ruleset_vars.items():
                ruleset_data[param] = var.get()

            # Generate a default name for the new ruleset
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            ruleset_name = f"custom_ruleset_{timestamp}"

            # Save the new ruleset
            success, result = save_ruleset(ruleset_name, ruleset_data, self.ruleset_data)
            if success:
                self.ruleset_data = load_ruleset_data()  # Refresh ruleset data
                self._refresh_ruleset_list()  # Update the ruleset dropdown
                messagebox.showinfo("Success", f"New ruleset '{result}' created successfully.")
                # Status message removed - using messagebox for important notifications only
                messagebox.showinfo("New Ruleset", f"New ruleset created: {result}")
            else:
                messagebox.showerror("Error", f"Failed to create new ruleset: {result}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while creating the ruleset: {e}")
            # Status message removed - using messagebox for important notifications only
            messagebox.showerror("Ruleset Error", f"Ruleset creation error: {e}")
            
    def _edit_selected_engine_ruleset(self):
        """Edit the selected engine ruleset."""
        selected_ruleset = self.ruleset_var.get()
        if not selected_ruleset:
            messagebox.showwarning("Warning", "No ruleset selected.")
            return

        if selected_ruleset in self.ruleset_data:
            self.current_ruleset = self.ruleset_data[selected_ruleset]
            self.ruleset_name_var.set(selected_ruleset)
            
            # Update the ruleset dropdown in the ruleset manager tab
            self.ruleset_dropdown_var.set(selected_ruleset)

            # Update editor fields
            for param, var in self.ruleset_vars.items():
                if param in self.current_ruleset:
                    var.set(self.current_ruleset[param])

            # Status message removed - using messagebox for important notifications only
            # Set the window title to indicate the ruleset being edited
            self.notebook.select(4)  # Switch to ruleset editor tab
        else:
            messagebox.showerror("Error", f"Ruleset '{selected_ruleset}' not found.")
            

def main():
    """Main entry point for the V7P3R Configuration GUI"""
    root = tk.Tk()
    app = ConfigGUI(root)
    
    # Set window icon if available
    try:
        icon_path = os.path.join(parent_dir, "images", "wK.png")
        if os.path.exists(icon_path):
            icon = tk.PhotoImage(file=icon_path)
            root.iconphoto(True, icon)
    except Exception as e:
        print(f"Could not load icon: {e}")
    
    # Center the window on the screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()