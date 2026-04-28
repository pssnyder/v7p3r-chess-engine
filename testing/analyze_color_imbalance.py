#!/usr/bin/env python3
"""
Analyze v19.5.6 tournament losses for patterns.

Tournament results (v19.5.6 perspective):
- Game 1: WIN (as White)
- Game 2: LOSS (as Black)  
- Game 3: WIN (as White, opponent timeout)
- Game 4: LOSS (as Black)
- Game 5: WIN (as White, opponent timeout)
- Game 6: LOSS (as Black)

Pattern: 3-0 as White, 0-3 as Black
This is a CRITICAL finding - suggests Black-side weakness!
"""

print("="*80)
print("V19.5.6 TOURNAMENT PATTERN ANALYSIS")
print("="*80)

results = [
    {"game": 1, "v19_color": "white", "result": "win"},
    {"game": 2, "v19_color": "black", "result": "loss"},
    {"game": 3, "v19_color": "white", "result": "win"},
    {"game": 4, "v19_color": "black", "result": "loss"},
    {"game": 5, "v19_color": "white", "result": "win"},
    {"game": 6, "v19_color": "black", "result": "loss"},
]

white_record = {"wins": 0, "losses": 0, "draws": 0}
black_record = {"wins": 0, "losses": 0, "draws": 0}

for game in results:
    if game["v19_color"] == "white":
        if game["result"] == "win":
            white_record["wins"] += 1
        elif game["result"] == "loss":
            white_record["losses"] += 1
        else:
            white_record["draws"] += 1
    else:
        if game["result"] == "win":
            black_record["wins"] += 1
        elif game["result"] == "loss":
            black_record["losses"] += 1
        else:
            black_record["draws"] += 1

print("\nCOLOR-BASED PERFORMANCE")
print("="*80)

white_games = white_record["wins"] + white_record["losses"] + white_record["draws"]
white_score = white_record["wins"] + white_record["draws"] * 0.5
white_pct = (white_score / white_games * 100) if white_games > 0 else 0

black_games = black_record["wins"] + black_record["losses"] + black_record["draws"]
black_score = black_record["wins"] + black_record["draws"] * 0.5
black_pct = (black_score / black_games * 100) if black_games > 0 else 0

print(f"\nAs WHITE: {white_record['wins']}W-{white_record['losses']}L-{white_record['draws']}D")
print(f"  Score: {white_score}/{white_games} ({white_pct:.1f}%)")

print(f"\nAs BLACK: {black_record['wins']}W-{black_record['losses']}L-{black_record['draws']}D")
print(f"  Score: {black_score}/{black_games} ({black_pct:.1f}%)")

print("\n" + "="*80)
print("CRITICAL FINDINGS")
print("="*80)

if white_pct == 100 and black_pct == 0:
    print("""
🚨 SEVERE BLACK-SIDE WEAKNESS DETECTED

v19.5.6 has PERFECT performance as White (100%) but ZERO as Black (0%)!

This is identical to the v17.0 PV instant move bug that caused:
- 100% win rate as White
- 64% win rate as Black
- All 3 tournament losses were as Black

POSSIBLE CAUSES:
1. Color-dependent evaluation bug
2. Black opening book weakness
3. Defensive position misvaluation
4. Time management differs by color
5. PV instant move on Black's turn

RECOMMENDED ACTIONS:
1. Review PV instant move logic (was this re-introduced?)
2. Test same position from both sides
3. Check if Black evaluations are negated correctly
4. Verify time management is color-neutral
5. Test Black openings specifically

This explains the 50% win rate - we're only competitive with White!
""")
elif abs(white_pct - black_pct) > 30:
    print(f"""
⚠️  SIGNIFICANT COLOR IMBALANCE

White: {white_pct:.1f}%
Black: {black_pct:.1f}%
Delta: {abs(white_pct - black_pct):.1f}% difference

This suggests a color-dependent issue that needs investigation.
""")
else:
    print("✓ No significant color imbalance detected")

print("\n" + "="*80)
print("RECOMMENDED NEXT STEPS")
print("="*80)

print("""
Priority 1: FIX BLACK-SIDE WEAKNESS
---------------------------------
1. Create test comparing same position from both sides
2. Review evaluation negation (Black vs White perspective)
3. Check for any color-dependent code paths
4. Test Black opening responses
5. Verify no PV instant move is happening on Black's turn

Priority 2: IMPROVE MOVE ORDERING
-------------------------------
1. Profile search to find why we search 4-6x more nodes
2. Tune history heuristic weights
3. Review killer move effectiveness
4. Consider MVV-LVA for captures

Priority 3: EXTENDED VALIDATION
-----------------------------
1. Fix Black-side issue first
2. Re-run tournament with fix
3. Target 60%+ win rate (not just 50%)
4. Test against diverse opponents

Current assessment:
- v19.5.6 has competitive search depth ✓
- v19.5.6 has good NPS ✓
- v19.5.6 has CRITICAL Black-side bug ✗
- v19.5.6 has move ordering inefficiency ⚠️

DO NOT DEPLOY until Black-side issue is resolved!
""")
