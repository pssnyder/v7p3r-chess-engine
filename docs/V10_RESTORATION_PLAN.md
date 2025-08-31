# V7P3R V10 Feature Restoration Plan

## Current Status
✅ **Performance Achieved**: 15,000+ NPS with bitboard evaluation  
✅ **Clean Codebase**: No duplicated classes, clean structure  
✅ **Basic Search**: Alpha-beta, iterative deepening, basic move ordering  

## Features Currently DISABLED (Need to Add Back)
❌ **Transposition Table**: Disabled for performance testing  
❌ **Null Move Pruning**: Disabled for performance testing  
❌ **Late Move Reduction (LMR)**: Disabled for performance testing  
❌ **Advanced Move Ordering**: Simplified to captures vs non-captures  
❌ **Tactical Heuristics**: Pin, fork, skewer, discovered attack detection  
❌ **Piece Defense Logic**: Complex piece interaction analysis  
❌ **Advanced Endgame Logic**: Endgame-specific evaluation  
❌ **Quiescence Search**: Tactical sequence extension  

## Implementation Strategy

### Phase 1: Core Search Optimizations (High Impact, Low Risk)
1. **Transposition Table** - Add back with performance monitoring
2. **Advanced Move Ordering** - MVV-LVA, killer moves, history heuristic
3. **Late Move Reduction** - Reduce search depth for later moves

### Phase 2: Pruning Techniques (Medium Impact, Medium Risk)
4. **Null Move Pruning** - Skip turns to identify futile positions
5. **Futility Pruning** - Skip moves that can't improve alpha
6. **Delta Pruning** - Skip captures that can't improve position enough

### Phase 3: Tactical Features (High Impact, High Risk)
7. **Quiescence Search** - Extend search for tactical sequences
8. **Check Extensions** - Search deeper when in check
9. **Basic Tactical Recognition** - Simple pin/fork detection

### Phase 4: Advanced Evaluation (Very High Impact, Very High Risk)
10. **Bitboard-Based Tactical Detection** - Fast pin/fork/skewer using bitboards
11. **Advanced Piece Defense** - Bitboard-powered piece interaction
12. **Endgame Tablebase Integration** - Perfect endgame play

## Performance Targets
- **After Phase 1**: Maintain 12,000+ NPS (20% performance cost acceptable)
- **After Phase 2**: Maintain 10,000+ NPS (33% performance cost acceptable)  
- **After Phase 3**: Maintain 8,000+ NPS (47% performance cost acceptable)
- **After Phase 4**: Maintain 6,000+ NPS (60% performance cost acceptable)

## Implementation Notes
- Add ONE feature at a time
- Run performance benchmark after each addition
- If NPS drops too much, optimize that feature before continuing
- Keep backup versions for rollback
- Focus on bitboard implementations for speed

## Current V10 Baseline
- **Performance**: 15,000+ NPS
- **Features**: Basic alpha-beta, iterative deepening, simple move ordering
- **Evaluation**: Bitboard-based material + positional
- **File**: v7p3r.py (clean version)
