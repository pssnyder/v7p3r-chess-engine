"""
Handles loading, mutating, and validating rulesets for GA.
"""
import yaml
import random
import copy

class RulesetManager:
    def __init__(self, ruleset_path="v7p3r_engine/rulesets/rulesets.yaml"):
        self.ruleset_path = ruleset_path

    def load_ruleset(self, name="default_evaluation"):
        """Load a ruleset by name from rulesets.yaml."""
        with open(self.ruleset_path, "r") as f:
            all_rulesets = yaml.safe_load(f)
        return copy.deepcopy(all_rulesets.get(name, {}))

    def save_ruleset(self, ruleset, name="ga_optimized"):
        """Save a ruleset to rulesets.yaml under a new name."""
        with open(self.ruleset_path, "r") as f:
            all_rulesets = yaml.safe_load(f)
        all_rulesets[name] = ruleset
        with open(self.ruleset_path, "w") as f:
            yaml.dump(all_rulesets, f)

    def mutate_ruleset(self, ruleset, mutation_rate=0.2, mutation_strength=0.1):
        """Randomly mutate numeric values in the ruleset."""
        mutated = copy.deepcopy(ruleset)
        for key, value in mutated.items():
            if isinstance(value, (int, float)) and random.random() < mutation_rate:
                delta = (random.random() * 2 - 1) * mutation_strength * abs(value if value != 0 else 1)
                mutated[key] = type(value)(value + delta)
        return mutated

    def crossover_rulesets(self, ruleset1, ruleset2):
        """Combine two rulesets by randomly picking values from each."""
        child = {}
        for key in ruleset1:
            if key in ruleset2:
                child[key] = random.choice([ruleset1[key], ruleset2[key]])
            else:
                child[key] = ruleset1[key]
        for key in ruleset2:
            if key not in child:
                child[key] = ruleset2[key]
        return child

    def validate_ruleset(self, ruleset):
        """Check for penalty/bonus sign conventions and return warnings."""
        warnings = []
        for key, value in ruleset.items():
            if "penalty" in key.lower() and value > 0:
                warnings.append(f"Penalty '{key}' should be negative, got {value}")
            if "bonus" in key.lower() and value < 0:
                warnings.append(f"Bonus '{key}' should be positive, got {value}")
        return warnings
