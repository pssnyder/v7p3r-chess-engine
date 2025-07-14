# v7p3r Chess Engine Design and Evaluation Rule Statements  
This document serves as a living design and brainstorming document outlining the current engine's functional goals and expectations.

**Engine Configuration:**

* Color Setting: White or Black  
* Core Engine Type: v7p3r
* Search Algorithm: negamax
* Ruleset: default, custom
* Depth: 1-10
* Performance Controls
  * Efficiency: Alpha Beta Pruning, Move Ordering with max move limit, Depth Limiting
  * Accuracy: Opening Book
  * Risk: Critical Move Short Circuiting, Quiescence Checks, MVV-LVA Calculation
  * Game Outcome Strictness: Stalemate Prevention, Draw Avoidance

**Piece Values in centipawns:**

* King \= 20000 points  
* Queen \= 900 points  
* Rook \= 500 points  
* Bishop \= 325 points  
* Knight \= 300 points  
* Pawn \= 100 point

**Search Types:**

* Simple: find the best immediate eval based on checkmates, material count, score, and pst only, fallback search type
* Negamax: primary search type, fully featured search with all performance options available, main search loop: engine chooses best overall eval for each turn in its depth, inverting calculation perspectives each move
* Random: plays any legal move, for testing purposes and opponent config

**Engine Modules:**

* Core Modules  
  * Chess Game: pygame handler for game configuration, game state handling, and game output rendering, including game pgn recording and metrics module calls.
  * V7P3R Engine: engine handler for game awareness, move generation, position evaluation, and final move selection
  * Config: engine config handler for loading and managing engine configuration settings
* Move Selection and Position Evaluation Modules
  * Search: move search handler, centralized search control module, calls performance, risk, move tree iteration and move scoring modules
  * Negamax: primary search algorithm, handles move tree iteration
  * Scoring: move scoring and position evaluation module, handles overall scoring calulation, short circuit logic, and evaluation values, calls out to individual scoring and evaluation modules
  * Rules: position specific score modifiers, move validators, and decision making module for the engine, sets guidelines and weighting of evaluation scores
  * Move Ordering: move prioritization and legal move limiting for increased move selection speed and preliminary move pruning
  * Tempo Calculation: critical priority scoring, priority move selection, and move avoidance, handles game phase, game continuance, and game ending condition checking for game state awareness, checkmate attacks/threats, stalemate avoidance, and draw prevention, can result in immediate move selection or complete principal variation avoidance
  * Primary Scoring: after Tempo, first order priority scoring module, handles material count, material score, and calls to piece square table calculation and calls to mvv-lva capture and threat assessment modules
  * Secondary Scoring: second order scoring module, handles castling and tactical decision scoring 
  * Piece Square Tables: piece square table evaluation module, handles piece square table calculation and game phase detection
  * MVV-LVA: simple module for most-valuable-victim/least-valuable-attacker logic for basic capture and threat awareness
  * Quiescence: active/risky position identification, examines move risk to achieve quieter positions beyond max depth for additional safety

**Utilities:**

* PGN Watcher  
* Live Ruleset/Puzzle Tuner  
* Metrics (Game and Move Evaluation)  
* Stockfish Handler  
* Utilities Class  
  * Path tools  
  * Datetime tools  
  * Debug tools

**Basic Evaluation Scoring Rules:**  
Critical Short Circuits:

1. Checkmates: If the engine can play a mate within 5, then instantly return that principle variation. If the opponent has a checkmate during move search then exit that pv exploration and skip that move.  
2. Stalemates: If the engine discovers a stalemate within the current principle variation, then exit that pv and skip that move.

Primary Scores:

1. Material Count: Engine’s raw piece count.  
2. Material Score: Engine’s piece value score using base piece values.  
3. Piece Square Tables: Engine’s piece position score using piece-square tables.  
4. Captures: Engine’s capture move scores using MVV-LVA calculation.  
   

Secondary Scores:

1. Castling: If the engine has castling rights and the move is to castle, then add the current engines total material score to the evaluation, else if the engine has castling rights and the move is a king or rook move and is not castling, then subtract the engines total material score.

**Engine Logic:**

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
   9. The engine evaluates stalemates second, skipping stakemates.  
   10. The engine evaluates primary scoring components, short circuiting for high scores.  
   11. The engine evaluates secondary scoring components, returning combined pv evaluation scores.  
   12. The engine compares pv scores, using the selected search algorithm’s logic to determine the best pv.  
   13. If enabled, the engine performs a quiescence search to identify non-quiet positions beyond the current depth.  
   14. If enabled, the engine checks the move for strict draw prevention, selecting an alternative best move if draw conditions exist.  
   15. If the root move in the selected pv is a valid move and is legal on the root game board, then play the move. If the move is invalid or not legal, raise an exception and repeat move search using a simple search, evaluating only legal moves and their immediate position evaluation, returning the move with the highest immediate eval score.  
   16. The engine plays the move on the root board.  
   17. The engine then plays the opponents move based on the current configuration.  
   18. The engine repeats self move and opponent move selection until the game is resolved with a 1-0, 0-1, or ½-½ result.  
   19. The engine records game metrics, move metrics, evaluation metrics, and pgn snapshots of the active game throughout play. The pgn is available for copy and pasting into an external analysis board for examination. The metrics are dumped to a database at strategic times to ensure data coverage is complete, but performant.  
   20. The final game record, including evaluation scores, is recorded to a timestamped eval game pgn in the games directory for posterity.  
4. If the game count is greater than one, a new game is automatically started based on the configured settings.  
5. If multiple config file names are passed to the engine via a CONFIG\_NAME array instead of a string of one config name, then the engine will iterate through configs creating a game instance for each config and playing out the set number of games for each configuration.  
6. Any exceptions during game play should be raised. Only final move validation has a fallback move selection path, all other engine breaking exceptions will pause processing.