# V7P3R Chess Engine Test Suite

## Overview
A comprehensive test suite for the V7P3R chess engine, covering core functionality, performance, and specific module tests.

## Test Structure

### 1. Core Module Tests
- `test_search.py`: Search algorithms and decision making
- `test_score.py`: Position evaluation and scoring
- `test_time.py`: Time management and allocation
- `test_config.py`: Configuration management

### 2. Chess Logic Tests
- `test_rules.py`: Move generation and validation
- `test_pst.py`: Piece-square table evaluation
- `test_mvv_lva.py`: Move ordering and capture analysis

### 3. Performance Tests
- `test_performance.py`: Benchmarks and efficiency tests
  - Move generation speed
  - Evaluation performance
  - Search depth timing
  - Move ordering efficiency

## Test Coverage

### Search Module
- Negamax algorithm
- Alpha-beta pruning
- Quiescence search
- Principal variation
- Iterative deepening
- Time management integration

### Evaluation Module
- Material counting
- Piece positioning
- Pawn structure
- King safety
- Mobility assessment

### Move Generation
- Legal move validation
- Special moves (castling, en passant)
- Move ordering efficiency
- Capture analysis

### Performance Metrics
- Move generation speed
- Position evaluation time
- Search depth scalability
- Memory usage
- Beta cutoff efficiency

## Running Tests

### All Tests
```powershell
cd "s:\Maker Stuff\Programming\V7P3R Chess Engine\v7p3r_chess_engine"
python -m unittest discover testing/ -v
```

### Specific Modules
```powershell
python -m unittest testing/test_search.py -v
python -m unittest testing/test_performance.py -v
```

## Performance Benchmarks

### Target Metrics
1. Move Generation
   - < 1ms per position
   - < 0.1ms std deviation

2. Position Evaluation
   - < 5ms per position
   - < 0.5ms std deviation

3. Search Performance
   - Depth 1: < 5ms
   - Depth 2: < 50ms
   - Depth 3: < 500ms

4. Move Ordering
   - > 70% beta cutoffs
   - < 30% full-width searches

## Test Maintenance

### Adding New Tests
1. Create test file in `testing/` directory
2. Follow naming convention: `test_*.py`
3. Include setup and teardown methods
4. Add performance metrics where applicable

### Updating Tests
1. Maintain compatibility with current implementations
2. Update benchmark targets as needed
3. Document significant changes
4. Verify against different board states

## Future Improvements

1. Additional Test Coverage
   - Endgame tablebases
   - Opening book usage
   - Multi-threading support
   - Memory management

2. Performance Testing
   - CPU profiling
   - Memory profiling
   - Thread synchronization
   - Cache efficiency

3. Integration Testing
   - UCI protocol compliance
   - GUI interaction
   - External engine compatibility
