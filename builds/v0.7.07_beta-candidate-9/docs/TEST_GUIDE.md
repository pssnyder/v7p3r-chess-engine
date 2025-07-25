# v7p3r Chess Engine Test Guide

## Stockfish ELO Finder Simulation

This guide explains how to use the Stockfish ELO Finder simulation to determine the approximate ELO strength of your v7p3r chess engine configuration.

### Overview

The Stockfish ELO Finder is a specialized simulation type that automatically adjusts the ELO rating of the Stockfish opponent based on game results. This adaptive approach helps to:

1. Determine the approximate ELO strength of your current v7p3r configuration
2. Track ELO progression as you make improvements to the engine
3. Compare different engine configurations in terms of absolute playing strength

### How It Works

The simulation follows these steps:

1. v7p3r (playing as White) plays a series of games against Stockfish
2. After each game:
   - If v7p3r wins, Stockfish's ELO rating increases for the next game
   - If v7p3r loses, Stockfish's ELO rating decreases for the next game
   - If there's a draw, Stockfish's ELO increases slightly
3. The magnitude of ELO adjustments decreases over time as the simulation converges
4. The simulation stops when either:
   - The win rate stabilizes (convergence is reached)
   - The maximum number of games is played

At the end, the v7p3r's estimated ELO is calculated based on the final Stockfish ELO and the win/loss ratio.

### Setup Instructions

#### Option 1: Run from Command Line

1. Open a terminal or command prompt
2. Navigate to the v7p3r Chess Engine directory
3. Run the following command:

```powershell
python -c "from engine_utilities.adaptive_elo_simulator import AdaptiveEloSimulator; simulator = AdaptiveEloSimulator(); simulator.run_simulation()"
```

This will run the ELO finder with default settings.

#### Option 2: Using the Simulation Config

1. Edit the `simulation_config.yaml` file
2. Add a simulation using the "stockfish_elo_finder" template:

```yaml
simulations:
  - name: "Find ELO for Default v7p3r"
    template: "stockfish_elo_finder"
    config:
      initial_elo: 1500
      min_games_for_convergence: 20
      max_games: 100
    v7p3r:
      v7p3r:
        depth: 4
        ruleset: "aggressive_ruleset"
```

3. Run the simulation manager:

```powershell
python -m engine_utilities.game_simulation_manager
```

### Configuration Options

You can customize the ELO finder behavior with these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `initial_elo` | Starting ELO for Stockfish | 1500 |
| `min_elo` | Minimum ELO to test | 800 |
| `max_elo` | Maximum ELO to test | 3200 |
| `adjustment_factor` | Controls how aggressively ELO changes | 1.0 |
| `convergence_threshold` | Win rate stability threshold | 0.05 (5%) |
| `min_games_for_convergence` | Minimum games before checking convergence | 20 |
| `max_games` | Maximum number of games to play | 100 |

You can also override any v7p3r or game configuration parameters as needed.

### Example Test Scenarios

Here are some example scenarios to try:

#### 1. Basic ELO Finding

```yaml
simulations:
  - name: "v7p3r Basic ELO Finder"
    template: "stockfish_elo_finder"
    config:
      initial_elo: 1500
      max_games: 50
```

#### 2. Testing Different Search Depths

```yaml
simulations:
  - name: "v7p3r Depth-4 ELO Finder"
    template: "stockfish_elo_finder"
    v7p3r:
      v7p3r:
        depth: 4
  - name: "v7p3r Depth-5 ELO Finder"
    template: "stockfish_elo_finder"
    v7p3r:
      v7p3r:
        depth: 5
```

#### 3. Compare Evaluation Rulesets

```yaml
simulations:
  - name: "v7p3r Aggressive ELO Finder"
    template: "stockfish_elo_finder"
    v7p3r:
      v7p3r:
        ruleset: "aggressive_ruleset"
  - name: "v7p3r Conservative ELO Finder"
    template: "stockfish_elo_finder"
    v7p3r:
      v7p3r:
        ruleset: "conservative_ruleset"
```

### Understanding Results

After the simulation completes, you'll find the results in the `games` directory. Look for a file named like `elo_finder_YYYYMMDD_HHMMSS_results.yaml`. 

The key metrics to look at are:

- `v7p3r_estimated_elo`: The estimated ELO rating of your engine
- `final_elo`: The final Stockfish ELO that resulted in a roughly balanced match
- `win_rate`: The overall win rate achieved by your engine
- `elo_history`: How the Stockfish ELO changed throughout the simulation
- `converged`: Whether the algorithm reached a stable ELO estimate

### Interpreting ELO Ratings

Here's a rough guide to understanding ELO ratings:

| ELO Range | Player Strength |
|-----------|-----------------|
| 800-1200 | Beginner |
| 1200-1600 | Casual player |
| 1600-1800 | Club player |
| 1800-2000 | Strong club player |
| 2000-2200 | Expert |
| 2200-2400 | National master |
| 2400-2500 | International master |
| 2500+ | Grandmaster |
| 3000+ | Super-computer level |

### Troubleshooting

If you encounter any issues:

1. Check the log files in the `logging` directory
2. Ensure Stockfish is properly installed and accessible
3. Try with a lower initial ELO if the games are too one-sided
4. Increase `max_games` if convergence isn't being reached

### Additional Notes

- The ELO estimation becomes more accurate with more games
- Results may vary based on the specific positions encountered
- For more accurate results, consider running multiple ELO finder simulations and averaging the results
- The estimated ELO is calibrated against Stockfish and may not perfectly match other rating systems

For additional help or to report issues, please refer to the project repository.
