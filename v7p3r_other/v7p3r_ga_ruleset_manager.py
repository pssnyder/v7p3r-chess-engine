"""
v7p3r Genetic Algorithm Ruleset Manager
"""
import os
import json
from v7p3r_config import v7p3rConfig

# Path to the rulesets JSON file
RULESET_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'rulesets', 'custom_rulesets.json')


class v7p3rGARulesetManager:
    """
    Manages the custom_rulesets.json file for the GA tuner.
    """
    def __init__(self):
        # Ensure the rulesets directory exists
        os.makedirs(os.path.dirname(RULESET_PATH), exist_ok=True)

    def load_all_rulesets(self) -> dict:
        """
        Loads all rulesets from the JSON file.
        Returns an empty dict if the file does not exist.
        """
        try:
            with open(RULESET_PATH, 'r') as f:
                return json.load(f) or {}
        except FileNotFoundError:
            return {}

    def load_ruleset(self, name: str) -> dict:
        """
        Retrieves a single ruleset by name.
        """
        return self.load_all_rulesets().get(name, {})

    def save_ruleset(self, name: str, ruleset: dict):
        """
        Saves or updates a ruleset in the JSON file.
        """
        data = self.load_all_rulesets()
        data[name] = ruleset
        with open(RULESET_PATH, 'w') as f:
            json.dump(data, f, indent=4)