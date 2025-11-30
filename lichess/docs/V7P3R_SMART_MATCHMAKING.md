# V7P3R Smart Matchmaking System

## Overview
The V7P3R Smart Matchmaking system provides intelligent opponent selection based on historical game data analysis. Instead of random opponent selection, it strategically chooses opponents to maximize both ELO growth and data collection value.

## Features

### 1. Game History Analysis
- **Automatic PGN Parsing**: Analyzes all PGN files in `game_records/` directory
- **Opponent Statistics**: Tracks W/L/D ratios, rating differentials, last played dates
- **Performance Metrics**: Calculates win rates, identifies improvement targets
- **Recency Tracking**: Prioritizes opponents based on days since last game

### 2. Strategic Opponent Selection

The system uses three prioritized strategies:

#### Strategy 1: Improvement Targets (30% weight)
- Targets opponents we have losing records against (< 50% win rate)
- Focuses on close-rating opponents (-200 to +200 ELO difference)
- Prioritizes followed accounts for accountability
- **Goal**: Identify weaknesses and gather data for heuristic improvements

#### Strategy 2: Variety/Data Collection (50% weight)
- Challenges never-played or under-played opponents
- Ensures minimum game diversity (< 5 games with opponent)
- Avoids recent rematches (> 30 days since last game)
- **Goal**: Broaden dataset and avoid overfitting to specific opponents

#### Strategy 3: Priority Fallback (20% weight)
- Challenges designated rivals: `c0br4_bot`, `slowmate_bot`
- Ensures consistent benchmark comparisons
- Activates when other strategies fail or game count is low
- **Goal**: Maintain consistent baseline performance data

### 3. Slot Reservation
- **Reserve Challenge Slot**: Keeps one slot open for incoming challenges
- Ensures engine stays active with both outgoing and incoming games
- Configurable: `reserve_challenge_slot: true`

### 4. Followed Accounts Integration
- Fetches followed accounts via Lichess API
- Prioritizes followed opponents for improvement data
- Helps identify specific opponents for strategic practice

## Configuration

Add to `config.yml` under `matchmaking` section:

```yaml
matchmaking:
  # ... existing settings ...
  
  # V7P3R SMART MATCHMAKING
  smart_matchmaking_enabled: true       # Enable intelligent opponent selection
  reserve_challenge_slot: true          # Reserve 1 slot for incoming challenges
  use_followed_accounts: true           # Prioritize followed accounts
  priority_opponents:                   # Fallback opponents
    - c0br4_bot
    - slowmate_bot
  min_games_per_opponent: 5             # Minimum games before "played enough"
  improvement_target_weight: 0.3        # 30% - Challenge losing matchups
  variety_weight: 0.5                   # 50% - Challenge under-played opponents
  priority_weight: 0.2                  # 20% - Challenge priority opponents
```

## How It Works

### Initialization
1. System loads when `smart_matchmaking_enabled: true`
2. Parses all PGN files in `game_records/` directory
3. Builds opponent database with statistics
4. Fetches followed accounts from Lichess API

### Opponent Selection Process
1. **Refresh Intelligence** (every 5 minutes):
   - Re-analyze game records for updated statistics
   - Re-fetch followed accounts list
   - Generate matchmaking report to logs

2. **Choose Strategy** (weighted random):
   - Roll dice: 0-30% = Improvement Target
   - Roll dice: 30-80% = Variety/Data Collection
   - Roll dice: 80-100% = Priority Fallback

3. **Filter Online Opponents**:
   - Get list of currently online bots
   - Apply rating filters from config
   - Remove blocked opponents
   - Check challenge history (avoid recent declines)

4. **Select Opponent**:
   - Apply strategy-specific logic
   - Log selection reasoning
   - Create challenge with standard time controls

### Matchmaking Report Example

```
============================================================
V7P3R MATCHMAKING INTELLIGENCE REPORT
============================================================

Total Opponents: 111
Total Games: 660
Overall Record: 354W / 221L / 66D (53.6%)

TOP IMPROVEMENT TARGETS (Losing Records):
------------------------------------------------------------
1. joshsbot: 94g, 20W/49L/22D (21.3% WR), Last 42d ago
2. plynder_r6: 52g, 19W/28L/5D (36.5% WR), Last 39d ago
3. plynder_r7: 43g, 13W/26L/4D (30.2% WR), Last 43d ago

MOST PLAYED OPPONENTS:
------------------------------------------------------------
1. joshsbot: 94g (21.3% WR)
2. mechasoleil: 57g (54.4% WR)
3. plynder_r6: 52g (36.5% WR)

PRIORITY OPPONENTS STATUS:
------------------------------------------------------------
c0br4_bot: 7g, 57.1% WR, Last played 53d ago
slowmate_bot: Never played
============================================================
```

## Testing Locally

### 1. Test Intelligence Analyzer
```bash
cd lichess
python lib/v7p3r_matchmaking_intelligence.py game_records
```

This will:
- Parse all PGN files
- Display opponent statistics
- Show improvement targets
- Identify under-played opponents

### 2. Test Smart Matchmaking (Dry Run)
Enable in config but observe logs without actual challenges:
```yaml
matchmaking:
  smart_matchmaking_enabled: true
  allow_matchmaking: false  # Disable actual challenges for testing
```

Monitor logs for:
- "Using V7P3R Smart Matchmaking system"
- "V7P3R MATCHMAKING INTELLIGENCE REPORT"
- Strategy selections: "IMPROVEMENT TARGET strategy", "VARIETY strategy", etc.

### 3. Enable Production
```yaml
matchmaking:
  smart_matchmaking_enabled: true
  allow_matchmaking: true
```

## Deployment to Cloud

### Update Cloud Configuration
1. **Upload new config**:
   ```bash
   gcloud compute scp config.yml v7p3r-production-bot:/home/patss/config-new.yml \
     --zone=us-central1-a --project=v7p3r-lichess-bot
   ```

2. **Upload new modules**:
   ```bash
   # Create tarball with smart matchmaking modules
   tar -czf smart-matchmaking.tar.gz -C lib v7p3r_matchmaking_intelligence.py v7p3r_smart_matchmaking.py
   
   # Upload to VM
   gcloud compute scp smart-matchmaking.tar.gz v7p3r-production-bot:/tmp/ \
     --zone=us-central1-a --project=v7p3r-lichess-bot
   ```

3. **Deploy to container**:
   ```bash
   gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot --command="
     sudo docker exec v7p3r-production tar -xzf /tmp/smart-matchmaking.tar.gz -C /lichess-bot/lib/
     sudo mv /home/patss/config-new.yml /home/v7p3r/config.yml
     sudo chown v7p3r:v7p3r /home/v7p3r/config.yml
     sudo docker restart v7p3r-production
   "
   ```

4. **Verify deployment**:
   ```bash
   gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot --command="
     sudo docker logs v7p3r-production --tail=50
   "
   ```

Look for:
- "Using V7P3R Smart Matchmaking system"
- "Analyzed games against X unique opponents"
- Strategy logs showing intelligent opponent selection

## Benefits

### For ELO Growth
- **Targeted Improvement**: Focus on opponents we're struggling against
- **Rating Optimization**: Challenge close-rated opponents for maximum ELO gain
- **Avoid Stagnation**: Prevent repeatedly playing same easy opponents

### For Analytics
- **Diverse Dataset**: Ensure broad opponent coverage
- **Weakness Identification**: Track specific matchups we're losing
- **Benchmark Consistency**: Regular games against priority opponents

### For Development
- **Data-Driven Decisions**: Clear reports on what heuristics need improvement
- **Followed Account Practice**: Target specific opponents for focused improvement
- **Automated Testing**: Consistent benchmark opponents for version comparisons

## Monitoring

### Key Metrics to Watch
1. **Challenge Success Rate**: Are opponents accepting challenges?
2. **Win Rate Trends**: Is ELO growing over time?
3. **Opponent Diversity**: Are we playing varied opponents?
4. **Strategy Distribution**: Are all three strategies activating?

### Log Messages
- `IMPROVEMENT TARGET strategy: Challenging X` - Targeting weak matchups
- `VARIETY strategy: Challenging X` - Building diverse dataset
- `PRIORITY FALLBACK strategy: Challenging X` - Ensuring consistent benchmarks
- `All smart strategies failed` - May need config adjustment

## Troubleshooting

### No Improvement Targets Found
- **Cause**: Winning against all close-rated opponents (good problem!)
- **Solution**: Lower `improvement_target_weight`, increase `variety_weight`

### Too Many Challenges to Same Opponents
- **Cause**: Limited online bot pool
- **Solution**: Increase `opponent_rating_difference` in config

### Smart Matchmaking Not Loading
- **Check**: `smart_matchmaking_enabled: true` in config
- **Check**: Module files present in `lib/` directory
- **Check**: Logs for error messages during initialization

### No Priority Opponent Games
- **Cause**: Opponents not online or rating too different
- **Solution**: Add more priority opponents to list, widen rating difference

## Future Enhancements
- [ ] Time control preferences based on performance
- [ ] Variant-specific opponent selection
- [ ] Opening-based opponent targeting
- [ ] Real-time ELO projection for opponent selection
- [ ] Machine learning for optimal challenge timing
- [ ] Integration with training data for v7p3r-ai

## Architecture

```
lichess-bot.py
  └── lichess_bot.py (line 359)
       ├── Standard: matchmaking.Matchmaking
       └── Smart: v7p3r_smart_matchmaking.V7P3RSmartMatchmaking
                    └── v7p3r_matchmaking_intelligence.V7P3RMatchmakingIntelligence
                         ├── Parses: game_records/*.pgn
                         ├── Tracks: OpponentStats
                         └── Generates: Strategic opponent lists
```

## Credits
Developed for V7P3R Chess Engine v17.4+
Part of the V7P3R Performance & Analytics Enhancement Initiative
November 2025
