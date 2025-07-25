# v7p3r Neural Network (NN) Engine

The v7p3r Neural Network Engine is a machine learning-based chess engine that learns from historical games, improves through training, and leverages Stockfish analysis to enhance its positional understanding.

## Architecture

The v7p3r Neural Network Engine consists of the following components:

1. **Neural Network Model**: A convolutional neural network that evaluates chess positions
2. **Move Library**: A local SQLite database that stores positions, evaluations, and best moves
3. **Training Pipeline**: A system for learning from PGN games and Stockfish analysis
4. **Chess Game Integration**: Interface with the main chess_game.py for gameplay

## Configuration

Configuration is stored in `config/v7p3r_nn_config.yaml` and includes:

- **Training Parameters**: Batch size, learning rate, epochs, etc.
- **Model Architecture**: Network structure, hidden layers, dropout rate
- **Storage Settings**: Model and metrics saving options
- **Stockfish Analysis**: Depth, positions per game, time settings
- **Move Library**: Database path, confidence thresholds, pruning settings

## Using the Engine

### In the Chess Game UI

The NN engine can be selected for either white or black player in the chess game:

1. Edit `config/chess_game_config.yaml`
2. Set `white_engine` or `black_engine` to `v7p3r_nn`

Example:
```yaml
white_player:
  is_human: false
  engine: v7p3r_nn
  
black_player:
  is_human: true
```

### Command Line Usage

The neural network engine can be used directly from the command line:

```bash
# Train the model on PGN files
python v7p3r_nn_engine/v7p3r_nn.py --train --pgn games/game1.pgn games/game2.pgn

# Analyze a game with Stockfish to add evaluations to the move library
python v7p3r_nn_engine/v7p3r_nn.py --analyze games/game1.pgn --stockfish path/to/stockfish
```

## Training Process

The neural network learns through these methods:

1. **PGN Training**: Extracts positions and evaluations from game records
2. **Stockfish Analysis**: Analyzes positions to provide ground truth evaluations
3. **Self-Play**: The engine can play against itself to generate training data

## Move Library

The move library stores:
- Chess positions (FEN strings)
- Position evaluations
- Best moves for positions
- Source of each evaluation (neural network, Stockfish, PGN)
- Confidence scores for moves

## Neural Network Features

- Convolutional layers to capture spatial features of the board
- Position representation as 13 x 8 x 8 tensors (12 piece types + turn indicator)
- Tanh activation for evaluation output (-10 to 10 range)
- Capability to run on GPU (CUDA) if available

## Integration with Stockfish

The engine integrates with Stockfish to:
- Analyze positions for ground truth evaluations
- Validate and improve its own evaluations
- Learn strong moves for specific positions

## Future Improvements

Planned enhancements:
- Enhanced model architecture
- Position feature engineering improvements
- Opening book integration
- Endgame tablebases
- Reinforcement learning from self-play
