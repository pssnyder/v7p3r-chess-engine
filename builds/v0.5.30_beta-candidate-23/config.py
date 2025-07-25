# config.py
"""
Configuration settings for the chess engine and bot
"""

# Engine Settings
ENGINE_CONFIG = {
    'name': 'ChessBot',
    'version': '1.0',
    'author': 'Your Name',
    'max_depth': 8,
    'hash_size_mb': 64,
    'threads': 1,
}

# Lichess Bot Settings
LICHESS_CONFIG = {
    'accept_challenges': True,
    'min_time_seconds': 30,  # Minimum game time to accept
    'max_time_seconds': 7200,  # Maximum game time to accept (2 hours)
    'accept_variants': ['standard'],  # Only accept standard chess
    'accept_bots': True,  # Accept challenges from other bots
    'accept_humans': True,  # Accept challenges from humans
    'auto_chat_responses': True,  # Respond to common chat messages
}

# Time Management Settings
TIME_CONFIG = {
    'emergency_time_threshold': 10.0,  # Emergency time management below 10 seconds
    'time_multiplier_opening': 0.8,   # Use less time in opening
    'time_multiplier_middlegame': 1.0, # Normal time in middlegame
    'time_multiplier_endgame': 1.2,   # Use more time in endgame
    'minimum_move_time': 0.1,         # Always think at least 100ms
}

# Evaluation Settings
EVAL_CONFIG = {
    'use_opening_book': False,  # Currently not implemented
    'use_endgame_tablebase': False,  # Currently not implemented
    'contempt_factor': 0.0,  # Draw contempt (0 = neutral)
}

# Logging Settings
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': None,  # Set to filename to log to file
}
