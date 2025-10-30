# V14.5 Queen Blunder Investigation
**Date:** October 29, 2025  
**Game:** V14.5 vs Stockfish 1%, Move 26

## The Blunder

**Position after 25...Qxa4+:**
```
. . . . k . . r
p p . b b . p .
. . . . p . . .
. . . . . n . .
q . . P . . . .
. . r . . R . P
. . K . . . . .
. . . R . . . .
```

White is in check. Legal moves:
- `Kxc3` (captures rook on c3)
- `Kd2`
- `Kb2` ⚠️ **THE BLUNDER**
- `Kb1`

**What V14.5 Played:** `26. Kb2`

**Why It's a Blunder:**
- Black responds `26...Qb4+` (check!)
- White can only play `Ka2` or `Ka1`
- Black's queen on b4 is giving check AND completely safe
- No way to capture it, no way to block
- Position is lost

## Root Cause Analysis

### NOT a Blunder Firewall Issue
The blunder firewall checks if OUR pieces are hanging after OUR move. In this case:
- After Kb2, white's pieces are fine
- The problem is black's RESPONSE (Qb4+) creates a bad position
- This requires looking ahead 2 ply (1 full move)

### Actual Problem: Shallow Search Depth
From PGN: `26. Kb2 {(Kb2-a2 Qb4-b3+...) -65.41/5 9}`
- Search depth: **5 ply**
- Time used: **9 seconds** (suggests emergency time limit)
- The engine DID evaluate the position as -65.41 (terrible for white)
- But still played it - **why?**

### Hypothesis
Looking at the evaluation: `-65.41/5` means it saw depth 5 and knew it was bad. But the move was still played. Possibilities:

1. **All moves were bad** - Check if Kxc3, Kd2, Kb1 were evaluated as even worse
2. **Move ordering issue** - Kb2 was evaluated first, others cut off by time
3. **Emergency time limit** - Search stopped before fully exploring alternatives

## Time Management Context
- Game time control: 300+5 (5 minutes + 5 second increment)
- V14.5 has aggressive emergency time limits (85% threshold)
- Move 26 in a complex middlegame
- 9 seconds suggests emergency limit kicked in early

## V14.6 Solution
Phase-based evaluation aims to:
1. **Faster opening evaluation** → more time reserves for middlegame
2. **Deeper middlegame search** → catch 2-3 ply tactics like this
3. **Better NPS** → reach depth 6+ instead of depth 5

### Target Goals
- Opening: Depth 4-5 in 2-3 seconds (save time)
- Middlegame: Depth 6-7 in 8-10 seconds (tactical depth)
- Endgame: Depth 7-8 in 5-7 seconds (fewer pieces = deeper search)

## Action Items
1. ✅ Complete V14.6 phase-based refactor
2. ⏭️ Measure NPS improvements per phase
3. ⏭️ Test depth achievement in middlegame positions
4. ⏭️ Compare V14.6 vs V14.5 on this exact position
5. ⏭️ Consider relaxing emergency time limits in middlegame (allow more thinking)

## Lesson
This wasn't a blunder firewall failure - it was a **horizon effect**. The engine needs to search deeper to see multi-move tactics. V14.6's performance improvements should help reach the necessary depth to avoid these blunders.
