# V7P3R v5.1 Enhancement Plan
# Focus: Improved Capture Logic and Threat Detection

## Key Issues Identified from Tournament Results:
1. Poor tactical awareness (missing threats to own pieces)
2. Bad capture decisions (not considering consequences)
3. Vulnerable to skewers and discovered attacks
4. Repetitive moves instead of improving position
5. Not evacuating threatened pieces

## Planned Improvements for v5.1:

### 1. Enhanced Move Ordering
- Add threat detection for own pieces
- Prioritize moves that save threatened pieces
- Better capture evaluation using SEE (Static Exchange Evaluation)
- Detect and prioritize tactical motifs (pins, skewers, forks)

### 2. Improved Quiescence Search
- Better capture filtering using full SEE analysis
- Include checks that lead to tactics
- Consider defensive moves in quiescence
- Extend search for forcing moves

### 3. Threat Assessment System
- Detect when pieces are under attack
- Identify pinned and skewered pieces
- Calculate hanging piece penalties
- Prioritize moves that address immediate threats

### 4. Enhanced Capture Logic
- Full SEE implementation for accurate capture evaluation
- Consider all defenders and attackers in sequence
- Account for discovered attacks after captures
- Better endgame capture handling

### 5. Tactical Pattern Recognition
- Basic pin detection
- Simple skewer recognition
- Fork identification
- Discovered attack awareness

## Implementation Priority:
1. Enhanced threat detection (highest priority)
2. Improved capture evaluation with SEE
3. Better move ordering for threatened pieces
4. Enhanced quiescence search
5. Basic tactical pattern recognition
