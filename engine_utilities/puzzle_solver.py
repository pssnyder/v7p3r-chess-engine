# engine_utilities/puzzle_solver.py
"""
Puzzle Solver Implementation Details - For Copilot...
Overview
Develop a robust performance benchmarking suite within a new engine_benchmark.py located in the engine_utilities directory. This suite will evaluate the v7p3r chess engine's performance across various configurations by running the adaptive ELO finder (to determine the engines game playing ELO vs stockfish) then adds a new ability and method for solving puzzles from the project's puzzle library. The end result of this puzzle solving will be estimating the engine's current puzzle solving ELO rating vs its Stockfish equivalency ELO rating, and an additional goal of supporting detailed ELO data tracking for future analysis.

Learning Capability
Additionally during any puzzle solving the engine should add puzzle solutions along with poor moves to its persistent transposition libraries that it can use during games, this is equivalent to a human studying and remembering positions they encountered during puzzle solving and how to solve them, if the engine is correct when solving a puzzle it can remember the move, weighted with a positive evaluation score to make it more likely to be chosen during a game scenario. However, if the engine is wrong during puzzle solving, the puzzle solver should not provide the correct answer, instead the engine will need to store these non-best moves in a persistent "anti_transposition_move" table that informs the engine what moves are not the best move, and thusly anti-transpositions should be weighted with a negative evaluation score to make them less likely to be chosen.

Transposition Table Coupling and Cohesion
The use of these transposition tables is linked to the in-game-memory transposition table, however that table being built per game is only a temp table for quicker access, and should be leveraged first over the permanently stored move and anti-move transposition tables, especially in timed games. The temp in-game transposition table, if enabled, will only be available during game, if enabled, moves will be added to it, along with the permanent transposition table (the permanent table only getting periodic updates to save on processing and increase performance, such as only updating once a certain number of additional moves, (maybe 25-50?) have been added to the in-game transposition table). The usage of the permanent transposition_move and anti_transposition_move tables should be configurable options in the engines config, next to the existing transposition table enablement option (the existing setting being whether or not to enable the addition of new transposition moves to the engines move library in temp and permanent transposition tables, the other transposition_move config setting being whether or not to enable usage of the permanent transposition_move library, and the final anti_transposition_move config setting being whether or not to enable the anti_transposition_move library during move selection). The transposition_move library, if enabled, and the anti_transposition_move library, if enabled, will be used during games, during puzzle solving, and during any other activity challenging the engine in order to positively multiply the evaluation score and increase the likelihood the engine plays a best move if we already know it, or inversely to provide an additional negative score value to the current evaluation score of a move if we already know it's not the best move (it may not be bad but its not the best so we can keep searching for a better one), these transposition_move and anti_transposition_move enablement settings will not enable or disable any modifications to those tables, that can only happen if the temporary use_transposition_table setting is enabled which will enable the modifier functions which can go in and update the permanent table based on temporary information about moves and their evaluations that we discover during games, puzzle solving, etc. finally the temp in-game transposition table, if enabled, is only used as a manager for the permanent transposition_move and anti-transposition_move tables, and a first access move library during timed game events, otherwise move knowledge should come from the permanent transposition_move and anti_transposition_move libraries, if enabled. If all three settings are disabled, then no transposition moves should be stored or updated and no transposition_move or anti_transposition_move entries should be used for move selection.

Requirements
Benchmark Suite Location:

New module: engine_utilities/engine_benchmark.py.
this file will call the adaptive_elo_finder and the puzzle_solver, which can also both be run independently (low coupling high cohesion)
Puzzle-Based ELO Evaluation:

puzzle solver location: engine_utilities/puzzle_solver.py
puzzle directory: puzzles
puzzle input file: lichess_db_puzzle.csv
puzzle ingestion format: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
Puzzle ingestion should be performed on the csv file and the puzzles along with their solutions and all other data should be converted from the csv format in the file and ingested into a new puzzle database on our centralized storage. more puzzles can then be added to the centralized puzzle library as they become available. We will be able to dynamically add puzzles in any format by simply building a new converter and ingestion tool addon.
Dynamically select puzzles from the project’s puzzle library based on configurable ELO ranges (e.g., 900–1200, or custom ranges).
The puzzle_solver.py must serve puzzles of increasing difficulty and allow the engine three attempts per puzzle.
Upon a correct guess, the next puzzle should be of higher ELO, with increments becoming logarithmic at higher ratings to avoid skipping potentially solvable puzzles in lieu of puzzles the engine cannot solve. This elo climbing should be calculated similar to the adaptive elo finder to ensure the best fit and puzzle elo calculation
Configuration and Comparison:

Support running puzzle solving for multiple engine configurations in a single session for comparative analysis.
Associate each test result with its corresponding engine configuration, similar to how simulation results are linked to their configs.
Possibly just piggy back on existing simulation manager code to add in a new simulation type detault which redirects play over to the puzzle solver to start solving puzzles as configured.
Any puzzle related configurations will be found in config/puzzle_config.yaml
If use_transposition_table is disabled in settings then no moves should be added to the temporary in game transposition table and thus no permanent memory should be stored of new transposition_move entries nor anti_transposition_move entires (this is a useful option for testing when we do not want moves stored as results could report incorrectly and corrupt the engines clean move library)
Data Handling:

Record each puzzle attempt and result, then store the data locally as puzzle data results.
The local database manager processes these results as part of its data cleanup and reporting ETLs, in addition to simulation data for games that should already be processed.
If enabled, Transposition moves stored in temporary transposition tables should be added to permanent storage transposition_move and anti-transposition_move libraries/tables (ensuring no duplicates).
Output and Reporting:

For each puzzle run, output the best-performing configuration to a tested_configurations directory.
Output file should include the configuration file name, test run date/timestamp, ELO reached and type of test (e.g., v7p3r_20250617_2035_960_puzzle.yaml).
Ensure the output configuration is immediately usable as the core configuration for releases.
Ensure updates are being recorded to centralized transposition_move and anti_transposition_move libraries/tables.
Acceptance Criteria
 engine_utilities/engine_benchmark.py exists and implements the described functionality.
 Puzzle data import can be run on lichess_db_puzzle.csv
 Engine supports dynamic ELO-based puzzle selection with logarithmic rating increments.
 Puzzle Elo finding can be run with one or multiple configurations, and results are linked appropriately.
 Data is stored locally and processed by the local database manager for reporting.
 Best configuration outputs are generated in the correct format and location.
 In-game Transposition Table, permanent transpositon_move library and anti_transposition_move libraries are all operating as expected to improve move selection and provide more human like learning capabilities.
Rationale
This enhancement will provide a standardized and automated way to evaluate engine improvements, benchmark configurations, and streamline the release process by surfacing the best-performing setups for immediate use or further development. It also contributes to the engines overall performance by improving its move library available for quicker access during play.
"""