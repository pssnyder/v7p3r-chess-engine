# V7P3R v12.4 Castling Test Positions

## Test Cases: King Moves Instead of Castling

These positions were extracted from actual V7P3R v12.2 games where the engine moved its king manually instead of castling.

### Test Case 1: Engine Battle 20250926_3, Round 4 (White: V7P3R v12.2)
**Position before king move**: After 5...bxa6
```
FEN: r1bqk1nr/p2ppppp/1p2P3/8/8/8/PPPP1PPP/RNBQK1NR w KQkq - 0 6
```
**V7P3R played**: 6. Kf1 (manual king move)
**Should consider**: 6. O-O (kingside castling)

**PGN Context**:
```
1. Nc3 c6 2. Nf3 Nf6 3. e3 Ng4 4. Bd3 Na6 5. Bxa6 bxa6 6. Kf1
```

### Test Case 2: Engine Battle 20250926_3, Round 5 (White: V7P3R v12.2)
**Position before king move**: After 4...a6
```
FEN: rnbqkb1r/2pppp1p/p5pn/8/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 0 5
```
**V7P3R played**: 5. Kf1 (manual king move)
**Should consider**: 5. O-O (kingside castling)

**PGN Context**:
```
1. Nc3 h6 2. e4 Rh7 3. Nf3 b6 4. Bc4 a6 5. Kf1
```

### Test Case 3: Engine Battle 20250926_3, Round 8 (White: V7P3R v12.2)
**Position before king move**: After 3...Nf6
```
FEN: rnbqkb1r/pppp1ppp/4pn2/8/2B1P3/2N5/PPPP1PPP/R1BQK1NR w KQkq - 1 4
```
**V7P3R played**: 4. Kf1 (manual king move)
**Should consider**: 4. O-O (kingside castling)

**PGN Context**:
```
1. Nc3 e5 2. e4 Nc6 3. Bc4 Nf6 4. Kf1
```

### Test Case 4: Engine Regression Battle 20250926_2 (White: V7P3R v12.2)
**Position before king move**: After 4...f6
```
FEN: rnbqk1nr/ppppbppp/4p3/8/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 1 5
```
**V7P3R played**: 5. Kf1 (manual king move)  
**Should consider**: 5. O-O (kingside castling)

**PGN Context**:
```
1. Nc3 e6 2. Nf3 Bb4 3. e4 f6 4. Bc4 Nc6 5. Kf1
```

## Test Methodology

For each position:
1. **Set up the position** in v12.4 (current development version)
2. **Run search** for 5-10 seconds with normal depth
3. **Compare move choice**:
   - ❌ **Manual king move** (Kf1, Ke2, etc.) = Failed test
   - ✅ **Castling** (O-O, O-O-O) = Passed test
   - ✅ **Other developing move** = Acceptable (shows not prioritizing king moves)

## Expected V12.4 Improvements

With enhanced castling evaluation, V7P3R v12.4 should:
1. **Prefer castling** over manual king moves when castling is available
2. **Apply penalties** for moving king without castling
3. **Recognize castling value** in opening positions
4. **Improve move ordering** to consider castling early

## Test Results Template

### Position 1 Results:
- **V12.2**: Kf1 (❌ Manual king move)
- **V12.4**: [To be tested]
- **Status**: [PASS/FAIL]

### Position 2 Results:
- **V12.2**: Kf1 (❌ Manual king move)  
- **V12.4**: [To be tested]
- **Status**: [PASS/FAIL]

### Position 3 Results:
- **V12.2**: Kf1 (❌ Manual king move)
- **V12.4**: [To be tested]
- **Status**: [PASS/FAIL]

### Position 4 Results:
- **V12.2**: Kf1 (❌ Manual king move)
- **V12.4**: [To be tested]
- **Status**: [PASS/FAIL]

---

**Success Criteria**: V12.4 should prefer castling or other developing moves over manual king movement in at least 3/4 test positions.