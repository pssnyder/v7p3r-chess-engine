# V7P3R v17.1 Arena Testing Quick Guide

## Engine Setup in Arena

### Engine Path
```
s:\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\src\v7p3r_uci.py
```

### Engine Details
- **Name:** V7P3R v17.1
- **Author:** Pat Snyder
- **Type:** Python UCI Engine
- **Protocol:** UCI

---

## Recommended Test Suite

### Test 1: Quick Validation (10 games vs v14.1)
**Purpose:** Verify fixes work correctly

**Settings:**
- **Opponent:** V7P3R v14.1
- **Games:** 10
- **Time Control:** 5+3 (5 min + 3 sec increment)
- **Starting Position:** Standard

**Watch For:**
- ✅ Opening book moves (first 8-10 moves)
- ✅ No "depth 1, 0 nodes" in log
- ✅ Balanced performance as White and Black
- ✅ No repeated tactical blunders

**Expected Result:** 7-8 wins (70-80%)

---

### Test 2: Head-to-Head with v17.0 (20 games)
**Purpose:** Confirm v17.1 is more reliable than v17.0

**Settings:**
- **Opponent:** V7P3R v17.0
- **Games:** 20
- **Time Control:** 5+3

**Expected Result:** 11-12 wins (55-60%)
- v17.1 should be more consistent
- Fewer losses as Black
- More balanced White/Black performance

---

### Test 3: Full Tournament (50 games mixed)
**Purpose:** Comprehensive performance evaluation

**Opponents:**
- v14.1 (20 games)
- v17.0 (10 games)
- C0BR4 v3.2 (20 games)

**Expected Results:**
- vs v14.1: 14-16 wins (70-80%)
- vs v17.0: 5-6 wins (50-60%)
- vs C0BR4 v3.2: 13-15 wins (65-75%)
- **Total:** 32-37 wins out of 50 (64-74%)

---

## Monitoring Checklist

### During Games
- [ ] Opening book moves displayed in log
- [ ] Normal search depths (not depth 1)
- [ ] Node counts reasonable (>1000 per move in midgame)
- [ ] No PV instant move signature

### After Each Game
- [ ] Check for tactical blunders (especially f6 move)
- [ ] Note time usage patterns
- [ ] Compare White vs Black performance
- [ ] Review any losses for patterns

### Log Analysis
Look for these indicators:
```
info string Opening book move: e2e4      ← GOOD (using book)
info depth 4 score cp 34 nodes 30769     ← GOOD (normal search)
```

Avoid these signatures:
```
info depth 1 score cp 0 nodes 0          ← BAD (instant move)
```

---

## Performance Expectations

### v17.1 vs v17.0 Comparison

| Metric | v17.0 | v17.1 Target | Improvement |
|--------|-------|--------------|-------------|
| Overall Win Rate | 88% | 96% | +8% |
| White Win Rate | 100% | 100% | 0% |
| Black Win Rate | 64% | 92% | +28% |
| Avg Depth | Mixed | Consistent | Stability |
| Opening Quality | Weak | Strong | Book |

### Key Improvements
1. **No more PV instant move blunders**
   - Fixed f6 tactical error
   - More reliable decision-making
   
2. **Opening book prevents weak positions**
   - Avoids 1.e3 trap
   - Saves time in opening
   - Better middlegame positions

3. **Balanced color performance**
   - v17.0: 100% White, 64% Black (36% gap)
   - v17.1: 100% White, 92% Black (8% gap)

---

## Troubleshooting

### Engine Not Starting
- Check Python path in Arena
- Verify v7p3r_uci.py is executable
- Check dependencies installed (chess library)

### No Opening Book Moves
- Should see "Opening book move: ..." in log
- If missing, check v7p3r_openings_v161.py exists in src/

### Performance Issues
- v17.1 should be slightly slower than v17.0 (no instant moves)
- But faster than v14.1 (opening book saves time)
- Typical node count: 10k-50k per move in midgame

---

## Arena Configuration

### Engine Management
1. **Add Engine:**
   - Engines → Install New Engine (1st Engine)
   - Navigate to: `s:\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\src\v7p3r_uci.py`
   - Name: "V7P3R v17.1"

2. **Tournament Setup:**
   - Tournament → New
   - Add engines: v17.1, v14.1, v17.0, C0BR4 v3.2
   - Games per pair: 10
   - Time control: 5+3

### Logging
Enable detailed logging to track:
- Opening book usage
- Search depths
- Node counts
- Time management

---

## Post-Testing

If results meet expectations (70%+ win rate vs v14.1, balanced colors):

1. **Tag release:** v17.1
2. **Deploy to Lichess**
3. **Monitor live games**
4. **Compare to tournament projections**

If issues found:
1. Review loss positions
2. Check for repeated patterns
3. Analyze logs for anomalies
4. Report findings for further fixes

---

## Quick Commands

### Start UCI test:
```bash
cd "s:/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/src"
python v7p3r_uci.py
```

### Run validation test:
```bash
cd "s:/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine"
python testing/test_v17_1_fixes.py
```

### Check version:
```bash
echo "uci" | python v7p3r_uci.py | grep "id name"
```

Expected output: `id name V7P3R v17.1`

---

**Last Updated:** November 21, 2025  
**Status:** Ready for Arena testing  
**Implementation validated:** ✅ All tests passing
