"""
v7p3r Genetic Algorithm Ruleset Manager
"""
import os
import yaml
from v7p3r_engine.v7p3r_config_gui import RULESET_PATH


class v7p3rGARulesetManager:
    """
    Manages the rulesets.yaml file for the GA tuner.
    """
    def __init__(self):
        # Ensure the rulesets directory exists
        os.makedirs(os.path.dirname(RULESET_PATH), exist_ok=True)

    def load_all_rulesets(self) -> dict:
        """
        Loads all rulesets from the YAML file.
        Returns an empty dict if the file does not exist.
        """
        try:
            with open(RULESET_PATH, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def load_ruleset(self, name: str) -> dict:
        """
        Retrieves a single ruleset by name.
        """
        return self.load_all_rulesets().get(name, {})

    def save_ruleset(self, name: str, ruleset: dict):
        """
        Saves or updates a ruleset in the YAML file.
        """
        data = self.load_all_rulesets()
        data[name] = ruleset
        with open(RULESET_PATH, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)