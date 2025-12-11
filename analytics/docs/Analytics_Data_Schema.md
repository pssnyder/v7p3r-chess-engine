# V7P3R Analytics - Data Schema Documentation
## Complete Data Collection & Storage Specification

**Version**: 1.0  
**Date**: December 4, 2025  
**Purpose**: Define all data collected, stored, and reported by the analytics system

---

## Data Collection Pipeline

```
Lichess API → PGN Download → Stockfish Analysis → Data Extraction → Storage → Reporting
```

---

## 1. Game Metadata (from Lichess API)

### Source
`https://lichess.org/api/games/user/v7p3r_bot?tags=true&clocks=true&evals=false&opening=true&literate=true`

### Fields Collected

```json
{
  "game_id": "string",           // Unique Lichess game ID
  "date": "ISO8601",             // Game start timestamp
  "event": "string",             // Event name (Rated Blitz, etc.)
  "site": "url",                 // Lichess game URL
  "white": "string",             // White player username
  "black": "string",             // Black player username
  "result": "string",            // 1-0, 0-1, 1/2-1/2
  "time_control": "string",      // e.g., "180+0" (3+0 blitz)
  "white_elo": "integer",        // White rating
  "black_elo": "integer",        // Black rating
  "white_rating_diff": "integer",  // Rating change for white
  "black_rating_diff": "integer",  // Rating change for black
  "opening_eco": "string",       // ECO code (e.g., "B02")
  "opening_name": "string",      // Opening name
  "termination": "string",       // Normal, Time forfeit, etc.
  "moves": "string"              // PGN movetext
}
```

### Storage Location
- **Raw PGN**: `analytics_reports/{year}/week_{num}_{date}/pgn/games.pgn`
- **Parsed JSON**: `analytics_reports/{year}/week_{num}_{date}/game_metadata.json`

---

## 2. Move-by-Move Analysis (from Stockfish)

### Analysis Parameters
- **Depth**: 20 (configurable)
- **Time per move**: 0.5 seconds (configurable)
- **Multi-PV**: 5 (to calculate top-5 alignment)

### Fields Collected Per Move

```json
{
  "move_number": "integer",      // 1, 2, 3, ...
  "ply": "integer",              // Half-move number (1, 2, 3, ...)
  "move_san": "string",          // e4, Nf3, O-O (Standard Algebraic Notation)
  "move_uci": "string",          // e2e4, g1f3, e1g1 (UCI format)
  "color": "string",             // "white" or "black"
  
  "evaluation": {
    "score_cp": "integer",       // Centipawn score after move
    "score_mate": "integer|null",  // Mate in N (null if not mate)
    "depth": "integer",          // Analysis depth reached
    "nodes": "integer",          // Nodes searched
    "time_ms": "integer"         // Analysis time
  },
  
  "best_move": {
    "move_san": "string",        // Best move in SAN
    "move_uci": "string",        // Best move in UCI
    "score_cp": "integer",       // Score if best move played
    "score_mate": "integer|null"
  },
  
  "top_moves": [                 // Top 5 moves for alignment calculation
    {"move": "string", "score_cp": "integer"},
    ...
  ],
  
  "eval_diff_cp": "integer",     // Centipawn loss (positive = worse)
  "classification": "string",    // See classification thresholds below
  
  "position": {
    "fen": "string",             // Position after move
    "material": {
      "white": "integer",        // Material count
      "black": "integer"
    },
    "phase": "string"            // "opening", "middlegame", "endgame"
  }
}
```

### Move Classification Thresholds

```python
# Configurable in .env or config
THRESHOLD_BEST = 10          # 0-10 cp: "best"
THRESHOLD_EXCELLENT = 25     # 10-25 cp: "excellent"
THRESHOLD_GOOD = 50          // 25-50 cp: "good"
THRESHOLD_INACCURACY = 100   # 50-100 cp: "inaccuracy"
THRESHOLD_MISTAKE = 200      # 100-200 cp: "mistake"
THRESHOLD_BLUNDER = 400      # 200-400 cp: "blunder"
# >400 cp: "critical_blunder" (??blunder)
```

### Storage Location
- **Per-game JSON**: `analytics_reports/{year}/week_{num}_{date}/games/{game_id}_analysis.json`
- **Aggregated**: Included in version/weekly reports

---

## 3. Tactical Theme Detection

### Themes Tracked

```json
{
  "castling": {
    "kingside_count": "integer",    // O-O moves
    "queenside_count": "integer",   // O-O-O moves
    "castling_timing": "integer"    // Average move # when castled
  },
  
  "pawn_structure": {
    "isolated_pawns": "integer",    // Count of isolated pawns
    "doubled_pawns": "integer",     // Doubled pawn instances
    "passed_pawns": "integer",      // Passed pawn count
    "pawn_breaks": "integer"        // Central pawn advances
  },
  
  "piece_coordination": {
    "bishop_pair": "boolean",       // Had bishop pair?
    "knight_outposts": "integer",   // Knight on outpost squares
    "rook_open_files": "integer",   // Rook on open file
    "rook_seventh_rank": "integer", // Rook on 7th/2nd rank
    "queen_rook_battery": "integer" // Q+R on same file/rank
  },
  
  "tactical_motifs": {
    "pins": "integer",              // Pin occurrences
    "skewers": "integer",           // Skewer occurrences
    "forks": "integer",             // Fork occurrences
    "discovered_attacks": "integer", // Discovered attacks
    "double_attacks": "integer"     // Multiple piece threats
  },
  
  "threats": {
    "mate_threats": [               // Move numbers with mate threats
      {"move": 23, "mate_in": 3},
      ...
    ],
    "material_threats": "integer",  // Pieces hanging/threatened
    "checkmates_missed": "integer"  // Mate in N missed
  },
  
  "endgame_themes": {
    "king_activity": "float",       // King centralization score
    "opposition": "integer",        // Opposition achieved
    "triangulation": "integer",     // Triangulation used
    "zugzwang": "integer"          // Zugzwang positions
  }
}
```

### Detection Methods
- **Castling**: Direct move detection (O-O, O-O-O)
- **Pawn Structure**: Board state analysis
- **Tactics**: Position comparison + eval changes
- **Endgame**: Piece count triggers (≤6 pieces)

### Storage Location
- **Per-game**: Embedded in game analysis JSON
- **Aggregated**: Theme frequency reports per week/version

---

## 4. Performance Metrics (KPIs)

### Game-Level Metrics

```json
{
  "game_id": "string",
  "result": "string",             // win/loss/draw
  "color": "string",              // white/black
  "opponent_rating": "integer",
  "rating_diff": "integer",       // Our rating change
  "termination": "string",        // Normal, Time forfeit, Abandoned, etc.
  
  "accuracy": {
    "average_cpl": "float",       // Average centipawn loss
    "top1_alignment": "float",    // % moves matching best (0-100)
    "top3_alignment": "float",    // % moves in top 3
    "top5_alignment": "float",    // % moves in top 5
    "acpl_opening": "float",      // CPL in opening (moves 1-15)
    "acpl_middlegame": "float",   // CPL in middlegame (moves 16-40)
    "acpl_endgame": "float"       // CPL in endgame (moves 40+)
  },
  
  "move_quality": {
    "best_moves": "integer",
    "excellent_moves": "integer",
    "good_moves": "integer",
    "inaccuracies": "integer",
    "mistakes": "integer",
    "blunders": "integer",
    "critical_blunders": "integer"
  },
  
  "time_management": {
    "average_move_time": "float",  // Seconds per move
    "time_pressure_moves": "integer", // Moves with <10s remaining
    "flagged": "boolean"           // Lost on time?
  }
}
```

### Aggregated Weekly Metrics

```json
{
  "week_id": "string",            // "2025-W49"
  "date_range": {
    "start": "ISO8601",
    "end": "ISO8601"
  },
  
  "games_played": "integer",
  "engine_version": "string",     // From version tracker
  
  "results": {
    "wins": "integer",
    "losses": "integer",
    "draws": "integer",
    "win_rate": "float",          // % (0-100)
    "draw_rate": "float"
  },
  
  "termination_breakdown": {
    "normal": "integer",
    "time_forfeit": "integer",
    "abandoned": "integer",
    "rules_infraction": "integer"
  },
  
  "ratings": {
    "average_opponent": "float",
    "rating_change": "integer",   // Net change for week
    "performance_rating": "float" // Calculated performance
  },
  
  "accuracy_stats": {
    "average_cpl": "float",
    "median_cpl": "float",
    "top1_alignment": "float",
    "best_game_cpl": "float",
    "worst_game_cpl": "float"
  },
  
  "blunder_analysis": {
    "total_blunders": "integer",
    "blunders_per_game": "float",
    "critical_blunders": "integer",
    "critical_per_game": "float",
    "most_common_blunder_position": "fen"
  },
  
  "theme_summary": {
    "castling_rate": "float",     // % games with castling
    "bishop_pair_games": "integer",
    "tactical_opportunities_found": "integer",
    "tactical_opportunities_missed": "integer"
  }
}
```

### Storage Location
- **Weekly**: `analytics_reports/{year}/week_{num}_{date}/weekly_summary.json`
- **Historical**: `analytics_reports/historical_summary.json`

---

## 5. Version Tracking

### Version Timeline

```json
{
  "versions": [
    {
      "version": "v17.5",
      "deployment_date": "2025-12-02T00:00:00Z",
      "notes": "Endgame optimization + castling pruning",
      "changes": [
        "PST adjustments for endgame",
        "Castling evaluation pruning in endgames"
      ],
      "performance_targets": {
        "top1_alignment": 47.0,
        "critical_blunders_per_game": 4.5
      }
    },
    ...
  ]
}
```

### Game → Version Mapping

**Logic**:
1. Parse game date from PGN
2. Find latest version where `deployment_date <= game_date`
3. Assign version to game metadata

### Storage Location
- **Version Config**: `analytics/version_timeline.json` (user-editable)
- **Per-game**: Version field in game metadata

---

## 6. Storage Architecture

### Directory Structure

```
analytics_reports/
├── 2025/
│   ├── week_48_2025-11-24/
│   │   ├── weekly_summary.json          # Aggregated metrics
│   │   ├── technical_report.md          # Human-readable
│   │   ├── games/
│   │   │   ├── abc123_analysis.json     # Individual game analysis
│   │   │   ├── def456_analysis.json
│   │   │   └── ...
│   │   ├── pgn/
│   │   │   └── v7p3r_weekly_2025-11-24.pgn
│   │   └── metadata.json                # Week metadata
│   └── week_49_2025-12-01/
│       └── ...
├── historical_summary.json              # All-time aggregated data
├── version_comparison.json              # Version-to-version metrics
└── schema_version.txt                   # Data schema version
```

### Historical Summary Structure

```json
{
  "schema_version": "1.0",
  "last_updated": "ISO8601",
  "total_weeks": "integer",
  "total_games": "integer",
  
  "by_version": {
    "v17.5": {
      "games": 223,
      "win_rate": 51.6,
      "avg_cpl": 1342,
      "top1_alignment": 47.3,
      "deployment_date": "2025-12-02",
      "weeks_active": 1
    },
    ...
  },
  
  "by_week": [
    {
      "week_id": "2025-W49",
      "version": "v17.5",
      "games": 223,
      "win_rate": 51.6,
      "avg_cpl": 1342,
      "blunders_per_game": 4.2
    },
    ...
  ],
  
  "trends": {
    "win_rate_trend": "improving",      // Last 4 weeks
    "cpl_trend": "improving",
    "blunder_trend": "improving"
  }
}
```

---

## 7. Data Schema Versioning

### Current Version: 1.0

**When to increment**:
- Adding new fields → Minor version (1.0 → 1.1)
- Changing field types → Major version (1.0 → 2.0)
- Removing fields → Major version

### Migration Support
- Each report includes `schema_version` field
- Parsers check version before loading
- Migration scripts in `analytics/migrations/`

---

## 8. Configuration & Customization

### Editable Configuration Files

**`.env`** - Runtime parameters:
```bash
DAYS_BACK=7
WORKERS=12
ANALYSIS_DEPTH=20
ANALYSIS_TIME_PER_MOVE=0.5
THRESHOLD_BLUNDER=400
```

**`version_timeline.json`** - Version tracking:
```json
{
  "versions": [...]  // Add new versions here
}
```

**`theme_detection_config.json`** - Theme parameters:
```json
{
  "endgame_piece_threshold": 6,
  "outpost_squares": ["c6", "d6", "e6", "f6", ...]
}
```

---

## 9. Data Access & Querying

### Python API

```python
from analytics_storage import AnalyticsStorage

storage = AnalyticsStorage("analytics_reports")

# Get historical summary
summary = storage.get_historical_summary()

# Get specific week
week_data = storage.get_week("2025-W49")

# Get version comparison
comparison = storage.compare_versions("v17.1", "v17.5")

# Get blunder patterns
blunders = storage.get_common_blunders(limit=10)
```

### CLI Access

```bash
# Query weekly stats
python analytics_query.py --week 2025-W49

# Compare versions
python analytics_query.py --compare v17.1 v17.5

# Export to CSV
python analytics_query.py --export csv --output stats.csv
```

---

## 10. Modification Workflow

### Adding New Data Fields

1. **Update Schema Doc** (this file):
   - Add field to appropriate section
   - Document data type and purpose
   - Increment schema version if needed

2. **Update Data Classes**:
   - Modify `v7p3r_analytics.py` dataclasses
   - Add field with default value for backwards compatibility

3. **Update Storage**:
   - Modify serialization in `report_generator.py`
   - Test backwards compatibility with old reports

4. **Update Analysis**:
   - Add extraction logic in `v7p3r_analytics.py`
   - Add to appropriate report sections

5. **Test & Document**:
   - Run test analysis
   - Update example outputs
   - Commit with clear description

### Example: Adding "Checkmate Pattern" Field

```python
# 1. In v7p3r_analytics.py
@dataclass
class ThemeDetection:
    ...
    checkmate_patterns: List[str] = field(default_factory=list)  # NEW

# 2. In detection logic
def _detect_checkmate_pattern(self, board, move):
    if board.is_checkmate():
        pattern = self._classify_checkmate(board)
        return pattern
    return None

# 3. In report_generator.py
def _format_themes(self, themes):
    ...
    md += f"- **Checkmate Patterns**: {', '.join(themes.checkmate_patterns)}\n"
```

---

## 11. Data Retention & Cleanup

### Retention Policy
- **Weekly Reports**: Keep all (long-term historical value)
- **Individual Game Analysis**: Keep 52 weeks (1 year)
- **Raw PGN Files**: Keep 12 weeks (3 months)

### Cleanup Script
```bash
# Run monthly
python cleanup_old_data.py --keep-weeks 52 --keep-pgn 12
```

---

## Summary

This schema provides:
- ✅ **Comprehensive data collection** from Lichess API + Stockfish
- ✅ **Tactical theme tracking** (castling, promotions, piece coordination, etc.)
- ✅ **Blunder analysis** with classification thresholds
- ✅ **KPI tracking** (W/L/D, ELO, accuracy, termination types)
- ✅ **Version tracking** with programmatic changelog
- ✅ **Modifiable schema** with clear documentation
- ✅ **Historical storage** for long-term analysis

**Next Steps**:
1. Review and approve schema
2. Implement weekly_pipeline_local.py using this schema
3. Test with sample games
4. Schedule automated runs

---

**Schema Version**: 1.0  
**Last Updated**: December 4, 2025  
**Maintainer**: Pat Snyder
