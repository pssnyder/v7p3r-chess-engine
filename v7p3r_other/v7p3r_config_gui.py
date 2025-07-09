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
from tkinter import ttk, messagebox, scrolledtext
import datetime
from tkinter.font import Font
import copy
from v7p3r_config import v7p3rConfig

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Default values used for new configurations
config_manager = v7p3rConfig()
DEFAULT_CONFIG = config_manager.get_config()

# Path for saved configurations
CONFIG_DIR = os.path.join(parent_dir, "configs")
os.makedirs(CONFIG_DIR, exist_ok=True)

# Path for rulesets file
RULESET_PATH = os.path.join(parent_dir, "configs", "rulesets", "custom_rulesets.json")
RULESET_TEMPLATE_PATH = os.path.join(parent_dir, "configs", "rulesets", "ruleset_template.json")

# Check if custom_rulesets.json exists, if not create it with default values
def ensure_rulesets_json_exists():
    """Ensure that custom_rulesets.json exists with at least a default evaluation ruleset"""
    os.makedirs(os.path.dirname(RULESET_PATH), exist_ok=True)
    if not os.path.exists(RULESET_PATH):
        # Create with default ruleset
        default_ruleset = {
            "default_ruleset": get_default_ruleset_values()
        }
        try:
            with open(RULESET_PATH, 'w') as f:
                json.dump(default_ruleset, f, indent=4)
            print(f"Created default custom_rulesets.json at {RULESET_PATH}")
        except Exception as e:
            print(f"Error creating default custom_rulesets.json: {e}")

# Call to ensure the rulesets file exists
ensure_rulesets_json_exists()

# Load ruleset options from custom_rulesets.json
def load_rulesets():
    try:
        with open(RULESET_PATH, 'r') as file:
            rulesets = json.load(file)
            return list(rulesets.keys()) if rulesets else ["default_ruleset"]
    except Exception as e:
        print(f"Error loading rulesets: {e}")
        return ["default_ruleset"]
# Get available engine options
def get_engine_options():
    # Default engines always available
    engines = ["v7p3r", "stockfish", "human"]
    
    # Check for RL engine
    try:
        from v7p3r_rl import v7p3rRLEngine
        engines.append("v7p3r_rl")
    except ImportError:
        pass
    
    # Check for GA engine
    try:
        from v7p3r_ga import v7p3rGeneticAlgorithm
        engines.append("v7p3r_ga")
    except ImportError:
        pass
    
    # Check for NN engine
    try:
        from v7p3r_nn import v7p3rNeuralNetwork
        engines.append("v7p3r_nn")
    except ImportError:
        pass
    
    return engines

# Configuration management functions
def list_configs():
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

def delete_config(config_name):
    """Delete a configuration file"""
    if not config_name.endswith('.json'):
        config_name += '.json'
    
    filepath = os.path.join(CONFIG_DIR, config_name)
    
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True, None
        else:
            return False, "File not found"
    except Exception as e:
        return False, str(e)

def validate_config(config_data):
    """Basic validation of configuration data"""
    required_keys = ["game_config", "engine_config", "stockfish_config"]
    
    # Check if all required keys exist
    for key in required_keys:
        if key not in config_data:
            return False, f"Missing required key: {key}"
    
    # Check game_config
    game_config = config_data.get("game_config", {})
    game_required_keys = ["game_count", "starting_position", "white_player", "black_player"]
    for key in game_required_keys:
        if key not in game_config:
            return False, f"Missing required game config key: {key}"
    
    # Check engine_config
    engine_config = config_data.get("engine_config", {})
    engine_required_keys = ["name", "version", "ruleset", "search_algorithm", "depth", "max_depth"]
    for key in engine_required_keys:
        if key not in engine_config:
            return False, f"Missing required engine config key: {key}"
    
    # Check stockfish_config if black_player or white_player is stockfish
    if game_config.get("white_player") == "stockfish" or game_config.get("black_player") == "stockfish":
        stockfish_config = config_data.get("stockfish_config", {})
        stockfish_required_keys = ["stockfish_path", "elo_rating", "skill_level", "debug_mode", "depth", "max_depth", "movetime"]
        for key in stockfish_required_keys:
            if key not in stockfish_config:
                return False, f"Missing required stockfish config key: {key}"

    return True, None

# Ruleset management functions
def load_ruleset_data():
    """Load all rulesets from custom_rulesets.json"""
    try:
        with open(RULESET_PATH, 'r') as file:
            rulesets = json.load(file)
            return rulesets or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        return {}

def save_ruleset(ruleset_name, ruleset_data, all_rulesets=None):
    """Save a ruleset to custom_rulesets.json"""
    # If no name provided, generate timestamped name
    if not ruleset_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        ruleset_name = f"custom_ruleset_{timestamp}"
    
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
            json.dump(all_rulesets, f, indent=4)
        return True, ruleset_name
    except Exception as e:
        return False, str(e)

def get_default_ruleset_values():
    """Get default values for a new ruleset from the ruleset template"""
    try:
        # Try to load from ruleset template
        if os.path.exists(RULESET_TEMPLATE_PATH):
            with open(RULESET_TEMPLATE_PATH, 'r') as f:
                template = json.load(f)
                return template
        else:
            # Fallback to hardcoded values if template doesn't exist
            return {
                "material_score_modifier": {
                    "modifier_id": "material_score_modifier",
                    "modifier_value": 0.8,
                    "modifier_name": "Material Weight",
                    "modifier_desc": " for material value",
                    "modifier_catagory": "material"
                },
                "pst_score_modifier": {
                    "modifier_id": "pst_score_modifier",
                    "modifier_value": 0.5,
                    "modifier_name": "Piece-Square Table Weight",
                    "modifier_desc": " for piece-square table values",
                    "modifier_catagory": "pst"
                },
                # Add more fallback modifiers as needed
                "tempo_modifier": {
                    "modifier_id": "tempo_modifier",
                    "modifier_value": 0.1,
                    "modifier_name": "Tempo",
                    "modifier_desc": " for maintaining initiative",
                    "modifier_catagory": "tempo"
                }
            }
    except Exception as e:
        return {}

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
        
        # Initialize form variables
        self._init_form_variables()
        
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
        
    def _init_form_variables(self):
        """Initialize all form variables with default values"""
        # Game configuration variables
        game_config = self.current_config.get("game_config", {})
        self.game_count_var = tk.IntVar(value=game_config.get("game_count", 1))
        self.starting_pos_var = tk.StringVar(value=game_config.get("starting_position", "default"))
        self.white_player_var = tk.StringVar(value=game_config.get("white_player", "v7p3r"))
        self.black_player_var = tk.StringVar(value=game_config.get("black_player", "stockfish"))
        
        # Engine configuration variables
        engine_config = self.current_config.get("engine_config", {})
        self.engine_version_var = tk.StringVar(value=engine_config.get("version", "1.0.0"))
        self.engine_color_var = tk.StringVar(value=engine_config.get("color", "white"))
        self.ruleset_var = tk.StringVar(value=engine_config.get("ruleset", "default_ruleset"))
        self.search_algorithm_var = tk.StringVar(value=engine_config.get("search_algorithm", "alphabeta"))
        self.depth_var = tk.IntVar(value=engine_config.get("depth", 3))
        self.max_depth_var = tk.IntVar(value=engine_config.get("max_depth", 5))
        
        # Stockfish configuration variables
        stockfish_config = self.current_config.get("stockfish_config", {})
        self.stockfish_path_var = tk.StringVar(value=stockfish_config.get("stockfish_path", ""))
        self.elo_var = tk.IntVar(value=stockfish_config.get("elo_rating", 1000))
        self.skill_var = tk.IntVar(value=stockfish_config.get("skill_level", 1))
        self.debug_var = tk.BooleanVar(value=stockfish_config.get("debug_mode", False))
        self.sf_depth_var = tk.IntVar(value=stockfish_config.get("depth", 3))
        self.sf_max_depth_var = tk.IntVar(value=stockfish_config.get("max_depth", 5))
        self.movetime_var = tk.IntVar(value=stockfish_config.get("movetime", 1000))
        
        # Form configuration name
        self.form_name_var = tk.StringVar()

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
        saved_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        
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
        
        # Create a paned window for preview and editor
        ruleset_paned = ttk.PanedWindow(self.ruleset_tab, orient=tk.HORIZONTAL)
        ruleset_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Ruleset preview
        preview_frame = ttk.LabelFrame(ruleset_paned, text="Ruleset Preview", padding=10)
        ruleset_paned.add(preview_frame, weight=1)
        
        self.ruleset_preview = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD, width=50, height=20)
        self.ruleset_preview.pack(fill=tk.BOTH, expand=True)
        self.ruleset_preview.config(state=tk.DISABLED)
        
        # Ruleset editor
        self.ruleset_edit_frame = ttk.LabelFrame(ruleset_paned, text="Ruleset Editor", padding=10)
        ruleset_paned.add(self.ruleset_edit_frame, weight=2)
        
        edit_controls = ttk.Frame(self.ruleset_edit_frame)
        edit_controls.pack(fill=tk.X, pady=5)
        
        self.ruleset_name_var = tk.StringVar()
        ttk.Label(edit_controls, text="Ruleset Name:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(edit_controls, textvariable=self.ruleset_name_var, width=30).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(edit_controls, text="Save Ruleset", command=self._save_current_ruleset).pack(side=tk.LEFT, padx=5)
        ttk.Button(edit_controls, text="Reset Form", command=lambda: self._on_ruleset_selected(None)).pack(side=tk.LEFT, padx=5)
        
        # Placeholder for the form - will be populated in _refresh_ruleset_form
        
        # Refresh the ruleset list
        self._refresh_ruleset_list()
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
        config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default_rulset.json')
        categories = {}
        descriptions = {}
        try:
            with open(config_path, 'r') as config_path:
                # Set categories
                ruleset_config = json.load(config_path)
                for rule in ruleset_config:
                    category = rule.get("modifier_category", "Uncategorized")
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(rule["modifier_id"])
                    descriptions[rule["modifier_id"]] = rule.get("modifier_desc", " for unknown reasons")
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from configuration file: {e}")
        
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
        configs = list_configs()
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
        # Load all rulesets
        self.ruleset_data = load_ruleset_data()
        self.rulesets = list(self.ruleset_data.keys())
        
        # Update the dropdown
        self.ruleset_combo['values'] = self.rulesets
        
        # Select the first one if available
        if self.rulesets:
            self.ruleset_dropdown_var.set(self.rulesets[0])
            self._on_ruleset_selected(None)
        else:
            # Clear the preview if no rulesets
            self.ruleset_preview.config(state=tk.NORMAL)
            self.ruleset_preview.delete(1.0, tk.END)
            self.ruleset_preview.insert(tk.END, "No rulesets available.")
            self.ruleset_preview.config(state=tk.DISABLED)
    
    def _run_chess_game(self):
        """Run a chess game with the current configuration"""
        # Check if there's a valid configuration
        if not self.current_config:
            messagebox.showwarning("Warning", "No configuration loaded. Please load or create a configuration first.")
            return
        
        # Validate the configuration
        valid, error = validate_config(self.current_config)
        if not valid:
            messagebox.showerror("Invalid Configuration", f"The current configuration is invalid: {error}")
            return
        
        try:
            # Create temporary config file
            temp_config_name = f"temp_config_{get_timestamp()}"
            temp_config_path = os.path.join(CONFIG_DIR, f"{temp_config_name}.json")
            with open(temp_config_path, 'w') as f:
                json.dump(self.current_config, f, indent=4)
            
            # Initialize the chess game with the configuration
            try:
                # Create and run a chess game instance
                from v7p3r_play import v7p3rChess
                game = v7p3rChess(temp_config_name)
                game.run()
                
                messagebox.showinfo("Success", "Chess game completed successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to run chess game: {e}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare chess game: {e}")
    
    def _load_selected_ruleset(self):
        """Load the selected ruleset into the engine configuration"""
        selected = self.ruleset_dropdown_var.get()
        if not selected:
            messagebox.showwarning("Warning", "No ruleset selected.")
            return
        
        # Update the engine config with the selected ruleset
        if "engine_config" in self.current_config:
            self.current_config["engine_config"]["ruleset"] = selected
            messagebox.showinfo("Success", f"Engine configuration updated to use ruleset: {selected}")
        else:
            messagebox.showwarning("Warning", "Could not update engine configuration. Please ensure a valid configuration is loaded.")
    
    def _on_config_selected(self, config_manager):
        """Handle configuration selection from dropdown"""
        selected = self.config_var.get()
        if selected:
            config_data, error = config_manager.load_config_from_file(selected)
            if config_data:
                self.preview_text.config(state=tk.NORMAL)
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(tk.END, json.dumps(config_data, indent=4))
                self.preview_text.config(state=tk.DISABLED)
            else:
                messagebox.showerror("Error", f"Failed to load configuration: {error}")
    
    def _on_ruleset_selected(self, event):
        """Handle ruleset selection from dropdown"""
        selected = self.ruleset_dropdown_var.get()
        if selected and selected in self.ruleset_data:
            ruleset_data = self.ruleset_data[selected]
            self.ruleset_preview.config(state=tk.NORMAL)
            self.ruleset_preview.delete(1.0, tk.END)
            
            # Format the ruleset for display
            json_text = json.dumps({selected: ruleset_data}, indent=4)
            self.ruleset_preview.insert(tk.END, json_text)
            self.ruleset_preview.config(state=tk.DISABLED)
            
            # Update current ruleset 
            self.current_ruleset = ruleset_data
            
            # Update form if it exists
            if hasattr(self, 'ruleset_form_frame') and self.ruleset_form_frame:
                self._refresh_ruleset_form()
    
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
    
    def _load_selected_ruleset_old(self):
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
    
    def _save_current_ruleset(self):
        """Save the current ruleset with changes from the form"""
        # Get the ruleset name
        ruleset_name = self.ruleset_name_var.get().strip()
        if not ruleset_name:
            messagebox.showwarning("Warning", "Please enter a ruleset name.")
            return
        
        # Create a new ruleset data structure
        new_ruleset = {}
        
        # Update values from the form
        for key, var in self.ruleset_vars.items():
            if key in self.current_ruleset and isinstance(self.current_ruleset[key], dict):
                # Copy the existing structure but update the value
                new_ruleset[key] = self.current_ruleset[key].copy()
                try:
                    # Try to get the value as a float
                    new_ruleset[key]["modifier_value"] = float(var.get())
                except (ValueError, TypeError):
                    # If conversion fails, use the string value
                    new_ruleset[key]["modifier_value"] = var.get()
            else:
                # For simple values or backward compatibility
                try:
                    new_ruleset[key] = float(var.get())
                except (ValueError, TypeError):
                    new_ruleset[key] = var.get()
        
        # Save the ruleset
        success, result = save_ruleset(ruleset_name, new_ruleset)
        
        if success:
            messagebox.showinfo("Success", f"Ruleset '{result}' saved successfully.")
            
            # Refresh the ruleset list and select the new one
            self._refresh_ruleset_list()
            self.ruleset_dropdown_var.set(result)
            self._on_ruleset_selected(None)
        else:
            messagebox.showerror("Error", f"Failed to save ruleset: {result}")
    
    def _new_ruleset(self):
        """Create a new ruleset based on the template"""
        # Load the default template
        self.current_ruleset = get_default_ruleset_values()
        
        # Generate a new name with timestamp
        timestamp = get_timestamp()
        new_name = f"custom_ruleset_{timestamp}"
        self.ruleset_name_var.set(new_name)
        
        # Update the preview
        self.ruleset_preview.config(state=tk.NORMAL)
        self.ruleset_preview.delete(1.0, tk.END)
        self.ruleset_preview.insert(tk.END, json.dumps({new_name: self.current_ruleset}, indent=4))
        self.ruleset_preview.config(state=tk.DISABLED)
        
        # Refresh the form with the new template
        self._refresh_ruleset_form()
        
        messagebox.showinfo("New Ruleset", "A new ruleset template has been created. Modify values as needed and click 'Save Ruleset' when done.")
    
    def _refresh_ruleset_form(self):
        """Refresh the ruleset form based on the current ruleset"""
        if not hasattr(self, 'ruleset_form_frame'):
            # Create the form frame if it doesn't exist
            self.ruleset_form_frame = ttk.Frame(self.ruleset_edit_frame)
            self.ruleset_form_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        else:
            # Clear existing form widgets
            for widget in self.ruleset_form_frame.winfo_children():
                widget.destroy()
        
        # Create a canvas for scrolling
        canvas = tk.Canvas(self.ruleset_form_frame, bg=self.colors["bg"])
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(self.ruleset_form_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create a frame inside the canvas for the form elements
        form_inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=form_inner, anchor=tk.NW)
        
        # Group modifiers by category
        categories = {}
        for key, modifier in self.current_ruleset.items():
            if isinstance(modifier, dict) and "modifier_catagory" in modifier:
                category = modifier["modifier_catagory"]
                if category not in categories:
                    categories[category] = []
                categories[category].append(key)
            else:
                # For backwards compatibility with old format
                if "other" not in categories:
                    categories["other"] = []
                categories["other"].append(key)
        
        # Create form elements for each category
        row = 0
        self.ruleset_vars = {}  # Clear previous variables
        
        for category, keys in sorted(categories.items()):
            # Create category header
            category_label = ttk.Label(
                form_inner, 
                text=category.replace("_", " ").title(),
                font=("Segoe UI", 12, "bold"),
                foreground=self.colors["header"]
            )
            category_label.grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(10, 5))
            row += 1
            
            # Create form elements for each modifier in the category
            for key in sorted(keys):
                if key in self.current_ruleset and isinstance(self.current_ruleset[key], dict):
                    modifier = self.current_ruleset[key]
                    name = modifier.get("modifier_name", key.replace("_", " ").title())
                    desc = modifier.get("modifier_desc", "")
                    value = modifier.get("modifier_value", 0.0)
                    
                    # Label with tooltip
                    label = ttk.Label(form_inner, text=f"{name}:")
                    label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
                    
                    # Create tooltip (simplified)
                    if desc:
                        self._create_tooltip(label, f"{name}{desc}")
                    
                    # Value entry field
                    var = tk.DoubleVar(value=float(value))
                    self.ruleset_vars[key] = var
                    entry = ttk.Entry(form_inner, textvariable=var, width=10)
                    entry.grid(row=row, column=1, padx=5, pady=2)
                    
                    # Optional slider for visual adjustment
                    scale = ttk.Scale(
                        form_inner, 
                        from_=-10.0, 
                        to=10.0, 
                        orient=tk.HORIZONTAL, 
                        variable=var,
                        length=150
                    )
                    scale.grid(row=row, column=2, sticky=tk.W, padx=5, pady=2)
                    
                    row += 1
                else:
                    # Handle old format or non-dict values
                    name = key.replace("_", " ").title()
                    value = self.current_ruleset.get(key, 0.0)
                    
                    label = ttk.Label(form_inner, text=f"{name}:")
                    label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
                    
                    # Value entry field
                    try:
                        var = tk.DoubleVar(value=float(value))
                    except (ValueError, TypeError):
                        var = tk.StringVar(value=str(value))
                    
                    self.ruleset_vars[key] = var
                    entry = ttk.Entry(form_inner, textvariable=var, width=15)
                    entry.grid(row=row, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
                    
                    row += 1
        
        # Update the scroll region
        form_inner.update_idletasks()
        canvas.config(scrollregion=canvas.bbox(tk.ALL))
        
        # Add mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def _create_tooltip(self, widget, text):
        """Create a simple tooltip for a widget"""
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            
            # Create tooltip window
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(
                self.tooltip, 
                text=text, 
                background=self.colors["text_bg"],
                foreground=self.colors["text_fg"],
                relief="solid", 
                borderwidth=1,
                wraplength=250,
                justify="left",
                padding=(5, 3)
            )
            label.pack()
        
        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
                
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
        
    def _delete_selected_config(self):
        """Delete the selected configuration file"""
        selected = self.config_var.get()
        if not selected:
            messagebox.showwarning("Warning", "No configuration selected.")
            return
        
        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Deletion", 
            f"Are you sure you want to delete the configuration '{selected}'?",
            icon=messagebox.WARNING
        )
        
        if confirm:
            success, error = delete_config(selected)
            if success:
                messagebox.showinfo("Success", f"Configuration '{selected}' was deleted.")
                self._refresh_config_list()
            else:
                messagebox.showerror("Error", f"Could not delete configuration: {error}")
    
    def _validate_json(self):
        """Validate the JSON configuration in the editor"""
        try:
            # Get the JSON from the editor
            json_text = self.json_editor.get(1.0, tk.END)
            config_data = json.loads(json_text)
            
            # Validate the config structure
            is_valid, error = validate_config(config_data)
            
            if is_valid:
                messagebox.showinfo("Validation", "Configuration is valid.")
                return True
            else:
                messagebox.showwarning("Validation", f"Invalid configuration: {error}")
                return False
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Invalid JSON format: {e}")
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Error validating configuration: {e}")
            return False
    
    def _save_json_config(self):
        """Save the JSON configuration from the editor"""
        try:
            # Get JSON from editor
            json_text = self.json_editor.get(1.0, tk.END)
            config_data = json.loads(json_text)
            
            # Validate the config
            is_valid, error = validate_config(config_data)
            if not is_valid:
                messagebox.showwarning("Validation Failed", f"Configuration is invalid: {error}")
                return
            
            # Get configuration name
            config_name = self.config_name_var.get().strip()
            if not config_name:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                config_name = f"config_{timestamp}"
                self.config_name_var.set(config_name)
            
            # Save the configuration
            success, result = save_config(config_name, config_data)
            if success:
                messagebox.showinfo("Success", f"Configuration saved as {result}")
                self.current_config = config_data
                self._refresh_config_list()
            else:
                messagebox.showerror("Error", f"Failed to save configuration: {result}")
        except json.JSONDecodeError as e:
            messagebox.showerror("JSON Error", f"Invalid JSON format: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")
    
    def _load_json_to_form(self):
        """Load the JSON configuration to the form fields"""
        try:
            # Get JSON from editor
            json_text = self.json_editor.get(1.0, tk.END)
            config_data = json.loads(json_text)
            
            # Validate the config
            is_valid, error = validate_config(config_data)
            if not is_valid:
                messagebox.showwarning("Validation Failed", f"Configuration is invalid: {error}")
                return
            
            # Update current config
            self.current_config = config_data
            
            # Update form with values from config
            self._update_form_from_config(config_data)
            
            # Switch to form tab
            self.notebook.select(self.form_tab)
            
            self._show_status("Configuration loaded to form")
            
        except json.JSONDecodeError as e:
            messagebox.showerror("JSON Error", f"Invalid JSON format: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")
    
    def _update_form_from_config(self, config_data):
        """Update the form fields with values from the configuration"""
        try:
            # Get config sections
            game_config = config_data.get("game_config", {})
            engine_config = config_data.get("engine_config", {})
            stockfish_config = config_data.get("stockfish_config", {})
            
            # Update Game Configuration
            self.game_count_var.set(game_config.get("game_count", 1))
            self.starting_pos_var.set(game_config.get("starting_position", "default"))
            
            # Set player selections if they are in the valid options
            white_player = game_config.get("white_player", "v7p3r")
            black_player = game_config.get("black_player", "stockfish")
            
            if white_player in self.engine_options:
                self.white_player_var.set(white_player)
            
            if black_player in self.engine_options:
                self.black_player_var.set(black_player)
            
            # Update Engine Configuration
            self.engine_version_var.set(engine_config.get("version", "1.0.0"))
            
            # Set engine color if valid
            engine_color = engine_config.get("color", "white")
            if engine_color in ["white", "black"]:
                self.engine_color_var.set(engine_color)
            
            # Set ruleset if it exists in available rulesets
            ruleset = engine_config.get("ruleset", "default_ruleset")
            if ruleset in self.rulesets:
                self.ruleset_var.set(ruleset)
            
            # Set search algorithm if valid
            search_algorithm = engine_config.get("search_algorithm", "alphabeta")
            valid_algorithms = ["lookahead", "minimax", "alphabeta", "quiescence", "iterative_deepening"]
            if search_algorithm in valid_algorithms:
                self.search_algorithm_var.set(search_algorithm)
            
            # Update numeric values
            self.depth_var.set(engine_config.get("depth", 3))
            self.max_depth_var.set(engine_config.get("max_depth", 5))
            
            # Update Stockfish Configuration
            self.stockfish_path_var.set(stockfish_config.get("stockfish_path", ""))
            self.elo_var.set(stockfish_config.get("elo_rating", 1000))
            self.skill_var.set(stockfish_config.get("skill_level", 1))
            self.debug_var.set(stockfish_config.get("debug_mode", False))
            self.sf_depth_var.set(stockfish_config.get("depth", 3))
            self.sf_max_depth_var.set(stockfish_config.get("max_depth", 5))
            self.movetime_var.set(stockfish_config.get("movetime", 1000))
            
        except Exception as e:
            messagebox.showerror("Error", f"Error updating form: {e}")