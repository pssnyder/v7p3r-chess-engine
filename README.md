<<<<<<< HEAD
# v7p3r Chess Engine - Product README

## Features (Streamlit App)

- **Play vs AI:** Make moves as White or Black, and the AI will respond.
- **AI Configuration:** Choose AI type (lookahead, minimax, negamax, random) and search depth.
- **FEN Input:** Set up any position by pasting a FEN string.
- **Position Evaluation:** Instantly evaluate any position with a single click.
- **Human-Readable Moves:** Select moves in standard chess notation (SAN).
- **Move History:** See the full move list in readable notation.
- **Board Visualization:** Interactive chessboard updates after each move.
- **No Installation Needed:** Deployable on [Streamlit Cloud](https://streamlit.io/cloud) for instant sharing.

---

## Firebase Backend

v7p3r Chess Engine now uses Firebase for backend storage, analytics, and data processing. Key features:

- **Cloud Storage**: Store PGN files, trained models, and evaluation data
- **Firestore Database**: Maintain game metadata, metrics, and model tracking
- **Authentication**: Optional user accounts for personalized experiences
- **Analytics**: Track engine performance and user engagement

For setup instructions, see [FIREBASE_SETUP.md](FIREBASE_SETUP.md)

---

## Quick Start

### Web Demo (Streamlit)

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the web app:

    ```bash
    streamlit run streamlit_app.py
    ```

### Local Metrics Dashboard

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the dashboard:

    ```bash
    python metrics/chess_metrics.py
    ```

### Stockfish ELO Finder

Determine the ELO strength of your v7p3r configuration:

1. Quick run with default settings:

   ```bash
   python run_elo_finder.py
   ```

2. Customize parameters:

   ```bash
   python run_elo_finder.py --initial-elo 1500 --v7p3r-depth 4 --v7p3r-ruleset aggressive_evaluation
   ```

3. For more options and advanced usage:

   ```bash
   python run_elo_finder.py --help
   ```

See [TEST_GUIDE.md](TEST_GUIDE.md) for detailed instructions and interpretation of ELO results.

---

## Significant File Overview

- `chess_game.py` — Core chess game logic and rules
- `v7p3r_scoring_calculation.py` — AI scoring and evaluation logic
- `chess_metrics.py` — Engine performance metrics dashboard
- `metrics_store.py` — Metrics database and logic
- `v7p3r.py` — Core chess engine logic
- `piece_square_tables.py` — Piece-square evaluation tables

- `config.yaml` — Engine and AI configuration
- `testing/` — Unit and integration tests for each module
- `games/` — Saved games, configs, and logs (for local/dev use)

---

## Testing

- Each main `.py` file has a corresponding `[module]_testing.py` in `testing/`.
- Run individual tests:

    ```bash
    python testing/metrics_store_testing.py
    ```

- Or run a suite (see `testing/launch_testing_suite.py` and `testing/testing.yaml`).

---

## Deployment

- **Web:** Deploy `streamlit_app.py` to [Streamlit Cloud](https://streamlit.io/cloud).
- **Local:** Run any module directly for advanced features and metrics.

---

## Limitations

- No Lichess/UCI integration in the web demo.
- Local metrics dashboard requires Python environment.
- AI vs AI and distributed/cloud database support are in development.

---

## Example Usage

- Play a game or analyze a position in the web app.
- Tune engine parameters and visualize results in the dashboard.
- Run tests to verify engine and metrics correctness.

---

## License

Open source — feel free to use and modify!

### Analytics ETL System

The v7p3r Chess Engine includes a robust ETL (Extract, Transform, Load) system for analytics:

1. Run the ETL process to transform raw game data into analytics-ready format:

    ```bash
    python -m engine_utilities.etl_processor
    ```

2. Set up scheduled ETL jobs:

    ```bash
    python -m engine_utilities.etl_scheduler --setup-local
    ```

3. Monitor ETL performance:

    ```bash
    python -m engine_utilities.etl_monitor --job-history
    ```

See [ETL System Documentation](docs/etl_system.md) for details on the analytics architecture.
=======
# v7p3r Chess Engine - Product README

## Features (Streamlit App)

- **Play vs AI:** Make moves as White or Black, and the AI will respond.
- **AI Configuration:** Choose AI type (lookahead, minimax, negamax, random) and search depth.
- **FEN Input:** Set up any position by pasting a FEN string.
- **Position Evaluation:** Instantly evaluate any position with a single click.
- **Human-Readable Moves:** Select moves in standard chess notation (SAN).
- **Move History:** See the full move list in readable notation.
- **Board Visualization:** Interactive chessboard updates after each move.
- **No Installation Needed:** Deployable on [Streamlit Cloud](https://streamlit.io/cloud) for instant sharing.

---

## Firebase Backend

v7p3r Chess Engine now uses Firebase for backend storage, analytics, and data processing. Key features:

- **Cloud Storage**: Store PGN files, trained models, and evaluation data
- **Firestore Database**: Maintain game metadata, metrics, and model tracking
- **Authentication**: Optional user accounts for personalized experiences
- **Analytics**: Track engine performance and user engagement

For setup instructions, see [FIREBASE_SETUP.md](FIREBASE_SETUP.md)

---

## Quick Start

### Web Demo (Streamlit)

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the web app:

    ```bash
    streamlit run streamlit_app.py
    ```

### Local Metrics Dashboard

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the dashboard:

    ```bash
    python metrics/chess_metrics.py
    ```

### Stockfish ELO Finder

Determine the ELO strength of your v7p3r configuration:

1. Quick run with default settings:

   ```bash
   python run_elo_finder.py
   ```

2. Customize parameters:

   ```bash
   python run_elo_finder.py --initial-elo 1500 --v7p3r-depth 4 --v7p3r-ruleset aggressive_evaluation
   ```

3. For more options and advanced usage:

   ```bash
   python run_elo_finder.py --help
   ```

See [TEST_GUIDE.md](TEST_GUIDE.md) for detailed instructions and interpretation of ELO results.

---

## Significant File Overview

- `chess_game.py` — Core chess game logic and rules
- `v7p3r_scoring_calculation.py` — AI scoring and evaluation logic
- `chess_metrics.py` — Engine performance metrics dashboard
- `metrics_store.py` — Metrics database and logic
- `v7p3r.py` — Core chess engine logic
- `piece_square_tables.py` — Piece-square evaluation tables

- `config.yaml` — Engine and AI configuration
- `testing/` — Unit and integration tests for each module
- `games/` — Saved games, configs, and logs (for local/dev use)

---

## Testing

- Each main `.py` file has a corresponding `[module]_testing.py` in `testing/`.
- Run individual tests:

    ```bash
    python testing/metrics_store_testing.py
    ```

- Or run a suite (see `testing/launch_testing_suite.py` and `testing/testing.yaml`).

---

## Deployment

- **Web:** Deploy `streamlit_app.py` to [Streamlit Cloud](https://streamlit.io/cloud).
- **Local:** Run any module directly for advanced features and metrics.

---

## Limitations

- No Lichess/UCI integration in the web demo.
- Local metrics dashboard requires Python environment.
- AI vs AI and distributed/cloud database support are in development.

---

## Example Usage

- Play a game or analyze a position in the web app.
- Tune engine parameters and visualize results in the dashboard.
- Run tests to verify engine and metrics correctness.

---

## License

Open source — feel free to use and modify!

### Analytics ETL System

The v7p3r Chess Engine includes a robust ETL (Extract, Transform, Load) system for analytics:

1. Run the ETL process to transform raw game data into analytics-ready format:

    ```bash
    python -m engine_utilities.etl_processor
    ```

2. Set up scheduled ETL jobs:

    ```bash
    python -m engine_utilities.etl_scheduler --setup-local
    ```

3. Monitor ETL performance:

    ```bash
    python -m engine_utilities.etl_monitor --job-history
    ```

See [ETL System Documentation](docs/etl_system.md) for details on the analytics architecture.
>>>>>>> 07a8bd8b88a40e25c3039c45e202a1c15bd0bce9
