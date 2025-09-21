# v7p3r Chess Engine Design

This document serves as a living design and brainstorming document outlining the current engine's functional goals and expectations.

### Engine Information
* Engine ID: V7P3R_v11.3
* Core Engine Name: v7p3r
* Engine Version: v11.3 (Built on V10.6 Proven Baseline)
* Search Algorithm: Alpha-Beta Negamax with Iterative Deepening
  * Pruning: Alpha-Beta with aspiration windows
  * Transposition Tables: 50,000 entries with Zobrist hashing
* Depth: 1-12 (adaptive based on time control)
* No Opening Book
* No Endgame Tablebase
* Time Management: Adaptive for different time controls (10+5, 2+1, bullet)
* Move Ordering: Advanced Multi-Stage
  * PV moves from transposition table
  * Killer moves (2 per depth)
  * History heuristic
  * MVV-LVA capture ordering
  * Tactical pattern bonus
  * Nudge system instant moves
* Quiescence Search: Capture and check extensions
* Scoring Components:
  * **Critical Detection:**
    * Checkmate Detection (within 5 moves)
    * Stalemate Detection and avoidance
    * Draw Prevention heuristics
  * **V11.3 Enhanced Heuristics:**
    * Draw penalty system
    * Enhanced endgame king evaluation
    * Move classification (development, consolidation, etc.)
    * King restriction ("closing the box")
    * Phase-aware evaluation weighting
  * **Core Evaluation:**
    * Material Count and Values
    * Bitboard-powered piece-square tables
    * Advanced pawn structure analysis
    * King safety with attack zone analysis
    * Castling rights and timing
  * **Tactical Analysis:**
    * Time-budget adaptive tactical pattern recognition
    * Pin, fork, skewer, discovered attack detection
    * Hanging piece identification
    * Double attack patterns
* Advanced Features:
  * **Nudge System:** Database of 439 proven move patterns for instant play
  * **PV Tracking:** Principal variation following with predicted board states
  * **Evaluation Caching:** Position hash-based caching system
  * **Search Extensions:** Check extensions and tactical position extensions
* Performance Optimizations:
  * Bitboard evaluation for speed (~217K NPS baseline)
  * Incremental move generation
  * Lazy evaluation with early cutoffs
  * Memory-efficient data structures
* UCI Compatible with full feature set

## Piece Values

Piece values are hard coded into the engine in centipawns:
* King \= 20000 points  
* Queen \= 900 points  
* Rook \= 500 points  
* Bishop \= 325 points  
* Knight \= 300 points  
* Pawn \= 100 point

## Search Types

Engine will use a primary search type with a fallback and a random function for testing:
* Negamax: primary search type, fully featured search with all performance options available, main search loop: engine chooses best overall eval for each turn in its depth, inverting calculation perspectives each move
* Simple: find the best immediate eval based on checkmates, material count, score, and pst only, fallback search type
* Random: plays any legal move, for testing purposes and opponent config

## Engine Modules

### Core Modules
* Config [required]: config handler for loading and managing game, engine, puzzle, metrics, and all other configuration settings
* Chess Game [required]: pygame handler for game configuration, game state handling, and game output rendering, including game pgn recording and metrics module calls. Leaves move calculation, scoring, and process intensive functions to the engine modules.
* V7P3R Engine [required]: engine handler for move examination, position evaluation, and final move selection (interface independent)

### Move Selection Handlers
* Search [required]: move search handler, centralized search control module, calls performance, risk, move tree iteration and move scoring modules
* Negamax [primary]: primary search algorithm, handles move tree iteration

### Position Evaluation Modules
* Scoring [required]: move scoring and position evaluation module, handles overall scoring calulation, short circuit logic, and evaluation values, calls out to individual scoring and evaluation modules
* Rules [required]: position specific score modifiers, move validators, and decision making module for the engine, sets guidelines and weighting of evaluation scores
* Tempo Calculation [critical]: critical priority scoring, priority move selection, and move avoidance, handles game phase, game continuance, and game ending condition checking for game state awareness, checkmate attacks/threats, stalemate avoidance, and draw prevention, can result in immediate move selection or complete principal variation avoidance
* Primary Scoring [primary]: after Tempo, first order priority scoring module, handles material count, material score, and calls to piece square table calculation and calls to mvv-lva capture and threat assessment modules
* Secondary Scoring [optional]: second order scoring module, handles castling and tactical decision scoring 
* Piece Square Tables [optional]: piece square table evaluation module, handles piece square table calculation and game phase detection
* MVV-LVA [optional]: simple module for most-valuable-victim/least-valuable-attacker logic for basic capture and threat awareness

### Performance and Accuracy Modules
* Move Ordering [optional]: move prioritization and legal move limiting for increased move selection speed and preliminary move pruning
* Book [optional]: Opening book containing basic openings to a max of 10 moves (London, Queens Gambit, Caro Kann, Scandinavian, French, Dutch, Vienna, and King's Indian)
* Quiescence [optional]: active/risky position identification, examines move risk to achieve quieter positions beyond max depth for additional safety
* Alpha Beta Pruning [optional]: search add-on feature to trim low scoring pv branches from the move search tree

## Engine Utility Modules

### Engine Handlers:
* v7p3r Handler (to control the interface between gameplay and the v7p3r engine, play_chess.py)
* Stockfish Handler (to activate the stockfish.exe engine for testing and as an opponent for the v7p3r engine)

### Game Monitoring
* PGN Watcher: an independent module to monitor the active_game.pgn file. provides more performant game monitoring, visuals should not be implemented in the engine
* Metrics: an independent module to handle metrics collection to identify performance and move selection score issues

### Performance Enhancement and Testing
* Live Ruleset/Puzzle Tuner (to batch test specific scoring actions on predetermined positions)
* Batch Game Analyzer (to batch compare v7p3r's previous game evaluations vs stockfish evaluations and identify critical scoring differences between the engines)

## Basic Evaluation Scoring Rules

### Critical Short Circuits:
1. Checkmates: If the engine can play a mate within 5, then instantly return that principle variation. If the opponent has a checkmate during move search then exit that pv exploration and skip that move.  
2. Stalemates: If the engine discovers a stalemate within the current principle variation, then exit that pv and skip that move.

### Primary Scores:
1. Material Count: Engine’s raw piece count.  
2. Material Score: Engine’s piece value score using base piece values.  
3. Piece Square Tables: Engine’s piece position score using piece-square tables, specific to game phase (opening, middlegame, endgame).  
4. Captures: Engine’s capture move scores using MVV-LVA calculation.  

### Secondary Scores:
1. Castling: If the engine has castling rights and the move is to castle, then add the current engines total material score to the evaluation, else if the engine has castling rights and the move is a king or rook move and is not castling, then subtract the engines total material score.
2. Tactics: If the engine has no critical or primary moves, if a pin, skewer, discovered attack, or the opponent has a hanging piece for safe attack, then increase that moves score over other non-tactical moves.
3. Capture to Escape Check: If the engine is placed in check and has a winning or equal value capture move available, then the engine should score the capture move higher than a king move to break the check.

## Engine Logic

1. The engine configures its components and modules according to the selected config file.  
2. The engine starts a chess game.  
3. Game loop:  
   1. The engine starts the search for a move based on the current board position.  
   2. If enabled, the engine assesses book moves.  
   3. Based on the search algorithm selected, the engine starts to evaluate principle variations via that search type.  
   4. If enabled, the engine orders legal moves and limits move exploration to a maximum move count.  
   5. If enabled, the engine limits nodes searched to a maximum node count based on average pv scores thus far, reducing the maximum node count inversely compared to the average of current pv scores (increased averages in explored pv scores results in decreased additional nodes needing to be explored.)  
   6. If enabled, iterative deepening dynamically sets the depth lower than the set depth for an initial search, then performs pv evaluation, short circuiting higher scores. If no scores meet a set threshold, then the depth is set to its standard depth and another pv evaluation is performed, short circuiting higher scores. If scores still do not meet the set threshold, the engine continues adding additional depth, until a move surpasses the threshold, or until the max depth setting is reached. (For simplicity, time management is not implemented currently.)  
   7. The engine iterates through legal moves in each potential principle variation, scoring their positions.  
   8. The engine evaluates checkmates first, short circuiting for checkmating positions, skipping checkmated positions.  
   9. The engine evaluates stalemates second, skipping stalemate pvs.  
   10. The engine evaluates primary scoring components, short circuiting for high scores.  
   11. The engine evaluates secondary scoring components, returning combined pv evaluation scores.  
   12. The engine compares pv scores, using the selected search algorithm’s logic to determine the best pv.  
   13. If enabled, the engine performs a quiescence search to identify non-quiet positions beyond the current depth.  
   14. If enabled, the engine checks the move for strict draw prevention, selecting an alternative best move if draw conditions exist.  
   15. If the root move in the selected pv is a valid move and is legal on the root game board, then play the move. If the move is invalid or not legal, raise an exception and repeat move search using a simple search, evaluating only legal moves and their immediate position evaluation, returning the move with the highest immediate eval score.  
   16. The engine plays the move on the root board.  
   17. The chess game then plays the opponents move based on the current configuration.  
   18. The game repeats this process, alternating v7p3r moves and opponent moves until the game is resolved with a 1-0, 0-1, or ½-½ result.  
   19. The engine records game metrics, move metrics, evaluation metrics, and pgn snapshots of the active game throughout play. The pgn is available for copy and pasting into an external analysis board for examination. The metrics are dumped to a database at strategic times to ensure data coverage is complete, but performant.
   20. As the game progresses, an active_game.pgn should be written after each move to preserve the current game state and allow the pgn watcher to visualize the game for the user.
   21. The final game record, including evaluation scores, is recorded to a timestamped eval game pgn in the games directory for posterity.  
4. If the game count is greater than one, a new game is automatically started based on the configured settings.  
5. If multiple config file names are passed to the engine via a CONFIG\_NAME array instead of a string of one config name, then the engine will iterate through configs creating a game instance for each config and playing out the set number of games for each configuration.  
6. Any exceptions during game play should be raised. Only final move validation has a fallback move selection path, all other engine breaking exceptions will pause processing.
