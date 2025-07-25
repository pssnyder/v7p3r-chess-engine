#!/usr/bin/env python3
"""
Configuration System Demonstration

This script demonstrates the new centralized JSON-based configuration system
for the v7p3r chess engine. It shows how to:

1. Load the default configuration
2. Use runtime overrides
3. Create engine-specific configurations
4. Access module-specific configurations
"""

from v7p3r_config import v7p3rConfig
import json


def main():
    print("=" * 60)
    print("v7p3r Chess Engine - Configuration System Demo")
    print("=" * 60)
    
    # 1. Basic Configuration Loading
    print("\n1. Loading Default Configuration:")
    config_manager = v7p3rConfig()
    print(f"   Γ£ô Configuration loaded from: {config_manager.config_path}")
    print(f"   Γ£ô Available config sections: {list(config_manager.get_config().keys())}")
    
    # 2. Module-Specific Configuration Access
    print("\n2. Module-Specific Configuration Access:")
    engine_config = config_manager.get_engine_config()
    print(f"   Γ£ô Engine depth: {engine_config.get('depth', 'Not set')}")
    print(f"   Γ£ô Engine search algorithm: {engine_config.get('search_algorithm', 'Not set')}")
    
    stockfish_config = config_manager.get_stockfish_config()
    print(f"   Γ£ô Stockfish ELO: {stockfish_config.get('elo_rating', 'Not set')}")
    print(f"   Γ£ô Stockfish skill level: {stockfish_config.get('skill_level', 'Not set')}")
    
    # 3. Runtime Overrides
    print("\n3. Runtime Configuration Overrides:")
    print("   Original engine depth:", config_manager.get_engine_config().get('depth'))
    
    # Apply override
    config_manager.set_override('engine_config.depth', 20)
    print("   After override (depth=20):", config_manager.get_engine_config().get('depth'))
    
    # Apply multiple overrides
    config_manager.set_override('stockfish_config.elo_rating', 2000)
    config_manager.set_override('stockfish_config.skill_level', 15)
    print("   Stockfish ELO after override:", config_manager.get_stockfish_config().get('elo_rating'))
    print("   Stockfish skill after override:", config_manager.get_stockfish_config().get('skill_level'))
    
    # 4. Engine-Specific Configuration Creation
    print("\n4. Engine-Specific Configuration Creation:")
    
    # Create NN-specific config
    nn_overrides = {
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': 50
    }
    nn_config = config_manager.apply_engine_specific_overrides('v7p3r_nn', nn_overrides)
    print("   Γ£ô Created NN-specific configuration")
    print(f"   Γ£ô NN batch size: {nn_config.get_v7p3r_nn_config().get('batch_size')}")
    print(f"   Γ£ô NN learning rate: {nn_config.get_v7p3r_nn_config().get('learning_rate')}")
    
    # Create GA-specific config
    ga_overrides = {
        'population_size': 50,
        'generations': 25,
        'mutation_rate': 0.15
    }
    ga_config = config_manager.apply_engine_specific_overrides('v7p3r_ga', ga_overrides)
    print("   Γ£ô Created GA-specific configuration")
    print(f"   Γ£ô GA population size: {ga_config.get_v7p3r_ga_config().get('population_size')}")
    print(f"   Γ£ô GA generations: {ga_config.get_v7p3r_ga_config().get('generations')}")
    
    # 5. Runtime Config Creation with Keyword Arguments
    print("\n5. Runtime Configuration Creation:")
    runtime_config = config_manager.create_runtime_config(
        engine_config__depth=12,
        engine_config__max_depth=18,
        stockfish_config__elo_rating=1800,
        game_config__game_count=5
    )
    print("   Γ£ô Created runtime configuration with overrides")
    print(f"   Γ£ô Runtime engine depth: {runtime_config.get_engine_config().get('depth')}")
    print(f"   Γ£ô Runtime max depth: {runtime_config.get_engine_config().get('max_depth')}")
    print(f"   Γ£ô Runtime Stockfish ELO: {runtime_config.get_stockfish_config().get('elo_rating')}")
    print(f"   Γ£ô Runtime game count: {runtime_config.get_game_config().get('game_count')}")
    
    # 6. Override Summary
    print("\n6. Configuration Override Summary:")
    overrides = config_manager.get_override_summary()
    if overrides:
        print("   Active overrides:")
        print(json.dumps(overrides, indent=6))
    else:
        print("   No active overrides")
    
    # 7. Module Configuration Access by Name
    print("\n7. Dynamic Module Configuration Access:")
    for module_name in ['engine', 'stockfish', 'game', 'nn', 'ga', 'rl']:
        try:
            module_config = config_manager.get_config_for_module(module_name)
            print(f"   Γ£ô {module_name.upper()} config loaded successfully")
        except ValueError as e:
            print(f"   Γ£ù {module_name.upper()} config failed: {e}")
    
    print("\n" + "=" * 60)
    print("Configuration system demonstration completed successfully!")
    print("=" * 60)
    
    # 8. Clear overrides demonstration
    print("\n8. Clearing Overrides:")
    print("   Engine depth before clearing:", config_manager.get_engine_config().get('depth'))
    config_manager.clear_overrides()
    print("   Engine depth after clearing:", config_manager.get_engine_config().get('depth'))
    
    print("\nΓ£ô All configuration operations completed successfully!")


if __name__ == "__main__":
    main()
