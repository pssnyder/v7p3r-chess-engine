﻿# RAW DATA EXAMPLES

## CSV files contain puzzles solutions and move sequences

csv_puzzle_filepath = "puzzles/puzzles.csv"  # Path to the CSV file containing puzzle solutions
csv_puzzle_headers = "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags"
csv_puzzle_example1 = "00sHx,q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17,e8d7 a2e6 d7d8 f7f8,1760,80,83,72,mate mateIn2 middlegame short,https://lichess.org/yyznGmXs/black#34,Italian_Game Italian_Game_Classical_Variation"
csv_puzzle_example2 = "00sJ9,r3r1k1/p4ppp/2p2n2/1p6/3P1qb1/2NQR3/PPB2PP1/R1B3K1 w - - 5 18,e3g3 e8e1 g1h2 e1c1 a1c1 f4h6 h2g1 h6c1,2671,105,87,325,advantage attraction fork middlegame sacrifice veryLong,https://lichess.org/gyFeQsOE#35,French_Defense French_Defense_Exchange_Variation"

## JSON files contain evaluation records from engine analysis

json_evaluation_filepath = "evaluations/evaluations.jsonl"  # Path to the JSON file containing evaluation records
json_evaluation_format = {          # json_evaluation_format describes the expected JSON structure
    "fen": "",                      #   "fen": the position FEN only contains pieces, active color, castling rights, and en passant square.
    "evals": [                      #   List of evaluations at different depths
        {                           #   Each evaluation contains:
            "knodes": 0,            #       "knodes": number of kilo-nodes searched by the engine
            "depth": 0,             #       "depth": depth reached by the engine
            "pvs": [                #       "pvs": list of principal variations
                {                   #       Each PV contains:
                    "cp": 0,        #           "cp": centipawn evaluation. Omitted if mate is certain.
                    "mate": None,   #           "mate": mate evaluation. Omitted if mate is not certain.
                    "line": ""      #           "line": principal variation, in UCI format.
                }
            ]
        }
    ]
}
json_evaluation_example = {
  "fen": "2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -",
  "evals": [
    {
      "pvs": [
        {
          "cp": 311,
          "line": "g2e4 f7f5 e4b7 c8b7 f2f3 b7f3 e1e6 d8h4 c2h2 h4g4"
        }
      ],
      "knodes": 206765,
      "depth": 36
    },
    {
      "pvs": [
        {
          "cp": 292,
          "line": "g2e4 f7f5 e4b7 c8b7 f2f3 b7f3 e1e6 d8h4 c2h2 h4g4"
        },
        {
          "cp": 277,
          "line": "f4g3 f7f5 e1e5 d8f6 a1e1 b7f7 g2c6 f8d8 d4d5 e6d5"
        }
      ],
      "knodes": 92958,
      "depth": 34
    },
    {
      "pvs": [
        {
          "cp": 190,
          "line": "h5h6 d8h4 h6g7 f8d8 f4g3 h4g4 c2e4 g4e4 g2e4 g8g7"
        },
        {
          "cp": 186,
          "line": "g2e4 f7f5 e4b7 c8b7 f2f3 b7f3 e1e6 d8h4 c2h2 h4g4"
        },
        {
          "cp": 176,
          "line": "f4g3 f7f5 e1e5 f5f4 g2e4 h7f6 e4b7 c8b7 g3f4 f6g4"
        }
      ],
      "knodes": 162122,
      "depth": 31
    }
  ]
}

## Evaluation Rule settings example yaml

### YAML formatted files define the game config and evaluation rules and their respective weights for the chess engine

yaml_config_filepath = "games/eval_game_%d%Y%m%d_%H%M%S.yaml"  # Path to the YAML file containing game config
evaluation_rules = {
    "checkmate_threats_modifier": 1000000.0,        # Bonus for checkmate threats
    "center_control_modifier": 0.25,        # Bonus per center square controlled
    "knight_activity_modifier": 0.1,        # Multiplier per square attacked by knight
    "bishop_activity_modifier": 0.15,       # Multiplier per square attacked by bishop
    "king_safety_modifier": 1.5,            # Bonus per pawn in king shield
    "piece_development_modifier": -0.5,         # Penalty for having under-developed minor pieces (increased)
    "check_modifier": 50.0,                 # Bonus for giving check (reduced)
    "castling_modifier": 5.0,               # Bonus for castling rights
    "en_passant_modifier": 1.0,             # Bonus for en passant opportunity
    "pawn_promotion_modifier": 5.0,         # Bonus for pawn promotion
    "passed_pawns_modifier": 1.0,            # Bonus for having an un-opposed pawn (increased)
    "piece_attacks_modifier": 2.0,          # Bonus for attacking hanging pieces
    "knight_count_modifier": 1.0,            # Bonus for having knight pair
    "castling_protection_modifier": 3.0,    # Bonus for keeping the right to castle
    "material_weight_modifier": 0.8,              # Material calculation impact on eval
    "piece_coordination_modifier": 0.5,     # Bonus for piece coordination
    "pawn_structure_modifier": -0.5,        # Penalty for having doubled pawns
    "pawn_weaknesses_modifier": -0.5,       # Penalty for having backward pawns
    "bishop_vision_modifier": 1.0,          # Bonus for the bishops having good board vision
    "tempo_modifier": 0.1,                  # Bonus for having the right to move
    "rook_coordination_modifier": 0.25,     # Bonus for having the rooks on the same rank
    "stalemate_modifier": -1000000.0,     # Penalty for stalemate situations
    "draw_modifier": -500000.0,           # Penalty for draw situations
    "piece_protection_modifier": -2.0,    # Penalty for undefended pieces
    "open_files_modifier": 0.3,              # Bonus for having rooks on open files
    "board_coverage_modifier": 0.1,         # Bonus for piece mobility
    "checkmate_move_bonus": 1000000,     # Bonus for finding a checkmate during move ordering scoring
    "check_move_bonus": 10000,           # Bonus for finding a check move during move ordering scoring
    "hash_move_bonus": 5000,             # Bonus for finding a hash move during move ordering scoring
    "capture_move_bonus": 4000,          # Bonus for finding a capture move during move ordering scoring
    "promotion_move_bonus": 3000,        # Bonus for finding a promotion move during move ordering scoring
    "killer_move_bonus": 2000,           # Bonus for finding a killer moves during move ordering scoring
    "history_move_bonus": 1000,          # Bonus for finding a historical move during move ordering scoring
    "counter_move_bonus": 1000           # Bonus for finding a strong countermove during move ordering scoring
}

## PGN files contain game results and move sequences
### PGN files are used to store chess games in a standard format, including metadata like event, site, date, players, and the moves played

pgn_game_filepath = "games/eval_game_%d%Y%m%d_%H%M%S.pgn"  # Path to the PGN file containing game results

Example PGN file content:
```bash
[Event "AI vs. AI Game"]
[Site "Local Computer"]
[Date "2025.06.06"]
[Round "#"]
[White "AI: v7p3r via minimax"]
[Black "AI: None via random"]
[Result "1-0"]

1. Nh3 { Eval: 0.14 } 1... h6 { Eval: 0.72 } 2. Nf4 { Eval: 1.37 } 2... b5
{ Eval: 2.02 } 3. Nc3 { Eval: 2.74 } 3... Ba6 { Eval: 2.74 } 4. Nd3
{ Eval: 2.74 } 4... Nc6 { Eval: 2.02 } 5. Nc5 { Eval: 2.02 } 5... Rc8
{ Eval: 2.02 } 6. Nxa6 { Eval: 3.82 } 6... Rb8 { Eval: 3.82 } 7. Nxb8
{ Eval: 7.97 } 7... d5 { Eval: 9.83 } 8. Nxc6 { Eval: 11.30 } 8... a6
{ Eval: 13.38 } 9. Nxd8 { Eval: 18.43 } 9... g6 { Eval: 20.51 } 10. Nxd5
{ Eval: 20.31 } 10... Rh7 { Eval: 24.74 } 11. Nc6 { Eval: 23.82 } 11... f6
{ Eval: 27.25 } 12. Nxc7+ { Eval: 25.48 } 12... Kd7 { Eval: 29.98 } 13. Nxb5
{ Eval: 26.35 } 13... Ke6 { Eval: 30.85 } 14. Nd8+ { Eval: 24.28 } 14... Kd5
{ Eval: 30.28 } 15. Nc7+ { Eval: 25.78 } 15... Ke4 { Eval: 30.13 } 16. d4
{ Eval: 25.27 } 16... h5 { Eval: 1000025.49 } 1-0
```

## CSV files contain static metrics on overall performance and individual game statistics
csv_all_game_metrics_filepath = "metrics/static_metrics.csv"
csv_all_game_metrics_header = "total_games,wins,losses,draws,trend_df"
csv_all_game_metrics_data = "96,25,6,63,datetime   result  win  loss  draw  cum_win  cum_loss  cum_draw\n0  2025-05-31 03:05:04  1/2-1/2    0     0     1        0         0         1"

## Log files contain engine evaluation logs
log_engine_evaluation_filepath = "logging/engine_eval.log"  # Path to the log file containing engine evaluation logs
