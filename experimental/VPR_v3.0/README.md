# VPR Pure Potential Engine v3.0

## Pure Potential Philosophy Implementation

Revolutionary chess AI based on piece potential rather than traditional material evaluation.

### Core Principles
1. **Piece value = attacks + mobility** (NO material assumptions)
2. **Focus ONLY on highest and lowest potential pieces**
3. **Assume imperfect opponent play** (not perfect responses)
4. **Preserve chaotic positions** through lenient pruning
5. **Break from traditional chess engine assumptions**

### Key Features
- Pure potential evaluation system
- Highest/lowest piece focus strategy
- Chaos preservation mechanics
- UCI protocol compatibility for Arena Chess GUI
- Configurable chaos threshold parameter

### Performance Characteristics
- Target: 12K+ NPS at 15+ ply depth
- Philosophy: "If a knight attacks 8 squares and can move to 8 positions freely, it has a score of 16. An undeveloped rook with 2 attacks has score of 2. We prioritize the knight, not traditional material values."

### Files
- `src/vpr.py` - Core pure potential engine implementation
- `src/vpr_uci.py` - UCI interface for Arena compatibility  
- `VPR_v3.0.bat` - Arena launcher with Python 3.13 support

### Arena Integration
Launch with `VPR_v3.0.bat` for Arena Chess GUI compatibility.

Engine responds to UCI commands:
- `uci` - Engine identification
- `position` - Set board position
- `go` - Search for best move
- `setoption name Chaos_Threshold value X` - Configure chaos preservation

### Version Notes
- v3.0: Pure potential implementation with Arena UCI support
- Based on user's original vision: "pieces with the most potential and pieces with the least potential"
- Maintains experimental version progression (v1.0 → v2.0 → v3.0)