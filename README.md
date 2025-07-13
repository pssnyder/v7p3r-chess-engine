# V7P3R Chess Engine

A high-performance chess engine written in Rust, featuring advanced search algorithms, efficient position evaluation, and comprehensive chess rule implementation.

## Features

### Core Chess Functionality
- **Complete chess rule implementation** including all special moves (castling, en passant, promotion)
- **Move generation** with pseudo-legal and legal move filtering
- **Position evaluation** using material balance, piece-square tables, and positional factors
- **Zobrist hashing** for efficient position identification and transposition tables
- **Three-fold repetition detection** and fifty-move rule enforcement

### Search Algorithm
- **Alpha-Beta pruning** with fail-soft implementation
- **Iterative deepening** for time management and move ordering
- **Transposition table** with configurable size (default 64MB)
- **Move ordering** using:
  - Hash moves from transposition table
  - MVV-LVA (Most Valuable Victim - Least Valuable Attacker) for captures
  - Killer moves heuristic
  - History heuristic
- **Quiescence search** to avoid horizon effect
- **Check extensions** for critical positions

### Performance Features
- **Bitboard representation** for efficient move generation
- **Magic bitboards** for sliding piece attacks
- **Parallel search** capabilities (multithreading)
- **Efficient memory management** with pre-allocated structures

### Interface Support
- **UCI (Universal Chess Interface)** protocol support
- **FEN string** parsing and generation
- **PGN move notation** support
- **Interactive CLI** for testing and analysis

## Installation

### Prerequisites
- Rust 1.70 or higher
- Cargo (comes with Rust)

### Building from Source
```bash
git clone https://github.com/yourusername/v7p3r_chess_engine.git
cd v7p3r_chess_engine
cargo build --release
```

The compiled binary will be available in `target/release/`

## Usage

### Command Line Interface
```bash
# Run the engine in UCI mode
./v7p3r_chess_engine

# Run with custom hash size (in MB)
./v7p3r_chess_engine --hash 128
```

### UCI Commands
The engine supports standard UCI commands:

```
uci                 # Initialize UCI mode
isready            # Check if engine is ready
position fen <fen>  # Set position from FEN
position startpos   # Set starting position
go depth <n>        # Search to depth n
go movetime <ms>    # Search for specified milliseconds
go infinite         # Search until "stop" command
stop                # Stop current search
quit                # Exit the engine
```

### Example UCI Session
```
uci
id name V7P3R Chess Engine
id author V7P3R Development Team
uciok

position startpos moves e2e4 e7e5
go depth 10
info depth 1 score cp 30 nodes 25 nps 25000 time 1 pv g1f3
info depth 2 score cp 0 nodes 125 nps 125000 time 1 pv g1f3 b8c6
...
bestmove g1f3
```

## Architecture

### Module Structure
- `board.rs` - Board representation and move generation
- `search.rs` - Search algorithms and move ordering
- `evaluation.rs` - Position evaluation functions
- `uci.rs` - UCI protocol implementation
- `moves.rs` - Move representation and validation
- `bitboard.rs` - Bitboard operations and magic generation
- `zobrist.rs` - Zobrist hashing for positions
- `transposition.rs` - Transposition table implementation

### Key Data Structures
- **Board**: Bitboard-based representation with piece positions
- **Move**: Compact move encoding (16 bits)
- **SearchInfo**: Search state and statistics
- **TranspositionTable**: Hash table for position caching

## Configuration

### Engine Options
- **Hash Size**: Transposition table size (1-1024 MB)
- **Threads**: Number of search threads (1-64)
- **Move Overhead**: Time buffer for network latency (0-5000 ms)

### Evaluation Parameters
The engine uses tuned parameters for:
- Material values (pawn=100, knight=320, bishop=330, rook=500, queen=900)
- Piece-square tables for positional bonuses
- King safety evaluation
- Pawn structure analysis

## Performance

### Benchmarks
- **Perft**: Validates move generation accuracy
- **Search Speed**: 1-5 million nodes per second (hardware dependent)
- **Evaluation Speed**: 10+ million positions per second

### Strength
- Estimated rating: 2000-2200 Elo (depending on time control)
- Tactical ability: Solves most combinations up to 8-10 ply
- Endgame knowledge: Basic endgame principles implemented

## Development

### Running Tests
```bash
cargo test              # Run all tests
cargo test perft        # Run perft tests only
cargo test --release    # Run tests in release mode
```

### Benchmarking
```bash
cargo bench            # Run performance benchmarks
```

### Code Style
The project follows Rust standard conventions:
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting

## Contributing

Contributions are welcome! Areas for improvement include:
- Enhanced evaluation functions
- Opening book support
- Endgame tablebase integration
- Neural network evaluation
- Additional search optimizations

Please submit pull requests with:
- Clear commit messages
- Test coverage for new features
- Performance benchmarks for optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Chess programming wiki for algorithms and techniques
- Stockfish team for inspiration and ideas
- Rust community for excellent tooling and libraries

## Contact

For questions, bug reports, or feature requests, please open an issue on GitHub.
