# V7P3R v15.0 Baseline Performance Results

**Test Date:** 2025-11-04 17:34:44

## Test Configuration

- **Engine:** V7P3R v15.0 (Clean Material Baseline)
- **Evaluation:** Pure material counting + bishop pair bonus
- **Search:** Alpha-beta with iterative deepening
- **Depth:** 8 (default), variable by position

## Position-by-Position Results

### Opening Position

**FEN:** `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`

| Metric | V15.0 |
|--------|-------|
| Move | g1h3 |
| Nodes | 7,963 |
| Time | 0.563s |
| NPS | 14,135 |

### Tactical - Fork Opportunity

**FEN:** `r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5`

| Metric | V15.0 |
|--------|-------|
| Move | c4d3 |
| Nodes | 47,852 |
| Time | 4.468s |
| NPS | 10,710 |

### Tactical - Pin

**FEN:** `r1bqkb1r/pppp1ppp/2n5/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5`

| Metric | V15.0 |
|--------|-------|
| Move | f3d4 |
| Nodes | 22,254 |
| Time | 1.828s |
| NPS | 12,172 |

### Middlegame - Complex

**FEN:** `r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 2 9`

| Metric | V15.0 |
|--------|-------|
| Move | d4c6 |
| Nodes | 55,782 |
| Time | 4.739s |
| NPS | 11,769 |

### Endgame - Pawn Race

**FEN:** `8/5k2/8/5P2/8/8/5K2/8 w - - 0 1`

| Metric | V15.0 |
|--------|-------|
| Move | f2g3 |
| Nodes | 2,213 |
| Time | 0.101s |
| NPS | 21,998 |

### Mate in 1

**FEN:** `r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4`

| Metric | V15.0 |
|--------|-------|
| Move | h5f7 |
| Nodes | 3,483 |
| Time | 0.199s |
| NPS | 17,498 |

### Mate in 2

**FEN:** `2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1`

| Metric | V15.0 |
|--------|-------|
| Move | g3g6 |
| Nodes | 41,630 |
| Time | 2.804s |
| NPS | 14,846 |

## Summary Statistics

| Metric | V15.0 |
|--------|-------|
| Total Nodes | 181,177 |
| Total Time | 14.702s |
| Average NPS | 12,323 |

## Analysis

This baseline establishes V15.0's performance with **pure material evaluation only**.

### Strengths
- Clean, simple codebase
- Fast search (no heuristic overhead)
- Good TT utilization

### Next Steps
1. Test against Material Opponent head-to-head
2. Compare with V12.6 baseline
3. Gradually add positional heuristics
4. Re-test after each addition

