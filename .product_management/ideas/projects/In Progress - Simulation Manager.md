Overview
We need to create a new game_simulation_manager.py that will serve as a game simulation manager for the engine. This manager will:

Handle and launch asynchronous parallel training tasks.
Allow configuring and launching multiple different game styles within a single simulation sequence.
Cycle through various chess game simulations, selecting different configurations for each new game until a configured limit is reached for that game type.
Dynamically integrate with the current configuration system, ensuring that existing functionality for singular simulations is preserved when asynchronous simulation is not in use.
Transition Details
The new manager will become the main launcher for AI vs AI simulations, replacing the need to run chess_game.py directly in those cases.
For human vs AI games, the original configuration workflow must remain unchanged. The human should still be able to choose color, engine, and configuration as before through the configuration system.
The simulation manager should be extensible to eventually:
Control local nodes and request additional worker processes (requirements TBD in a future project outline).
Support distributed/cloud-based simulations for scaling up when adding new AI types (e.g., neural network-based search, genetic, and reinforcement models, which will live alongside the existing move selection search algorithm options of deepsearch, simple_search, and thus will enable us to configure some hybrid approaches, requirements also TBD in a future project outline).
Data and Logging
Ensure the manager can document and log configurations for each simulation, enabling correlation with game outcomes.
As simulation volume increases, we must complete the transition to an efficient storage method for game results (not file-based), or this enhancement will overwhelm the filesystem. (this dependency will be completed prior to this work beginning however considerations should be made during testing to account for load testing to ensure the localhost data solution still functions in the cloud)
Questions for the Team & Answers
Are there preferred libraries or patterns for async task management in this codebase?
Not at this time, use whatever is best practice for large data simulation engines with an exponentially deepening decision tree.
Is there an existing logging or game result schema we should integrate with, or should we propose a new one?
The game must be recorded in official PGN format with headers, moves, evaluation scores from the engine as comments, and the game result displayed as 1-0, 0-1, or 1/2-1/2.
Are there any current blockers for moving away from file-based result storage?
No blockers, but there is a risk. The move to a database will solve performance, commit filesize, and diff filecount limitation issues. However, syncing fractured collections of game metrics from other computers running simulations will still rely on git pushes.
Any other requirements for local worker management, or should this be planned for a separate follow-up once requirements are clearer?
This should be planned in a separate follow up. Noted here in case it may impact current design choices.