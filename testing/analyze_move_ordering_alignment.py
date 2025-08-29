#!/usr/bin/env python3
"""
V7P3R v8.2 Move Ordering vs Evaluation Alignment Analysis
Analyzing the alignment between move ordering heuristics and evaluation priorities for V8.2
"""

import sys
import os
sys.path.append('src')

import chess
from v7p3r import V7P3RCleanEngine

def analyze_current_move_ordering():
    """Analyze the current V8.2 move ordering system"""
    print("🔍 CURRENT V8.2 MOVE ORDERING ANALYSIS")
    print("=" * 60)
    
    print("\n📋 Current Move Ordering Priority (from _score_move_enhanced):")
    print("1. Mate Detection: 200,000 points")
    print("   - Highest priority for checkmate moves")
    
    print("2. Captures (MVV-LVA): 100,000+ points")
    print("   - Victim value * 10 - Attacker value")
    print("   - Prioritizes high-value captures with low-value pieces")
    
    print("3. Promotions: 90,000+ points")
    print("   - Plus piece value of promotion piece")
    
    print("4. Killer Moves: 80,000+ points")
    print("   - Moves that caused alpha-beta cutoffs at same ply")
    
    print("5. Checks: 5,000 points")
    print("   - Any move that gives check")
    
    print("6. History Heuristic: Up to 4,000 points")
    print("   - Moves that historically caused cutoffs")
    
    print("7. Center Control: 100 points")
    print("   - Moves to d4, d5, e4, e5 squares")
    
    print("\n⚠️  AREAS NOW ADDRESSED IN V8.2:")
    print("✅ Explicit mate detection (200k priority)")
    print("✅ Multi-piece tactical detection")
    print("✅ Game phase contextual ordering")
    print("✅ Efficient pre-pruning")
    print("✅ King safety moves (when appropriate)")
    print("✅ Opening development prioritization")
    print("✅ Endgame king activity prioritization")
    
    print("\n⚠️  AREAS STILL TO ENHANCE:")
    print("- More sophisticated threat/defense evaluation")
    print("- Enhanced king safety zone detection")
    print("- Piece coordination bonuses")

def analyze_current_evaluation():
    """Analyze the current V8.2 evaluation components"""
    print("\n🎯 CURRENT V8.2 EVALUATION COMPONENTS")
    print("=" * 60)
    
    print("\nFrom v7p3r_scoring_calculation.py:")
    print("1. Material Score - Basic piece counting")
    print("2. King Safety - Defensive positioning around king")
    print("3. Piece Development - Knights/bishops out, rooks active")
    print("4. Castling Bonus - Incentive for king safety")
    print("5. Rook Coordination - Open files, seventh rank")
    print("6. Center Control - Control of central squares")
    print("7. Endgame Logic - King activity, pawn promotion")
    
    print("\n💡 EVALUATION PRIORITIES:")
    print("- Material advantage (captures)")
    print("- King safety (both defensive and attacking)")
    print("- Piece activity and development")
    print("- Positional factors (center, coordination)")
    print("- Endgame technique")

def analyze_alignment_issues():
    """Identify misalignments between move ordering and evaluation"""
    print("\n⚖️  ALIGNMENT ANALYSIS")
    print("=" * 60)
    
    print("\n✅ GOOD ALIGNMENTS:")
    print("- Captures: Both prioritize material gain")
    print("- Promotions: Both value piece value increases")
    print("- Center control: Both recognize central square importance")
    
    print("\n❌ MISALIGNMENTS:")
    print("1. KING SAFETY MISMATCH:")
    print("   - Evaluation: Strong focus on king safety")
    print("   - Move Ordering: Only checks (attacking), no defensive priority")
    
    print("2. THREAT ASSESSMENT GAP:")
    print("   - Evaluation: Considers king safety & piece coordination")
    print("   - Move Ordering: No threat detection or defensive prioritization")
    
    print("3. PIECE VALUE PRECISION:")
    print("   - Evaluation: Sophisticated piece-square and development bonuses")
    print("   - Move Ordering: Basic MVV-LVA, no tactical refinement")
    
    print("4. POSITIONAL VS TACTICAL:")
    print("   - Evaluation: Balanced positional + tactical")
    print("   - Move Ordering: Heavy tactical bias, light positional")
    
    print("5. ENDGAME AWARENESS:")
    print("   - Evaluation: Specific endgame logic")
    print("   - Move Ordering: No endgame-specific priorities")

def analyze_proposed_ordering(user_priorities):
    """Analyze the user's proposed move ordering priorities"""
    print("\n🚀 PROPOSED MOVE ORDERING ANALYSIS")
    print("=" * 60)
    
    print("\n📝 User's Proposed Priority Order:")
    for i, priority in enumerate(user_priorities, 1):
        print(f"{i}. {priority}")
    
    print("\n🎯 ALIGNMENT WITH CURRENT EVALUATION:")
    
    alignments = [
        ("Mates", "Perfect alignment - missing from current ordering"),
        ("Captures", "Good alignment - current MVV-LVA works well"),
        ("Threats/Defense", "Critical gap - evaluation emphasizes king safety"),
        ("Multi-piece attacks", "Missing - would complement piece coordination eval"),
        ("Value-based attacks", "Refinement of current MVV-LVA approach"),
        ("Defense moves", "Critical gap - evaluation has strong defensive components"),
        ("King attack squares", "Good alignment - evaluation prioritizes king safety"),
        ("King defense squares", "Good alignment - evaluation prioritizes king safety"),
        ("Promotions/Castling", "Good alignment - currently handled"),
    ]
    
    for move_type, alignment in alignments:
        print(f"- {move_type}: {alignment}")

def test_alignment_with_position():
    """Test current vs proposed ordering on a tactical position"""
    print("\n🧪 PRACTICAL ALIGNMENT TEST")
    print("=" * 60)
    
    engine = V7P3RCleanEngine()
    
    # Tactical position with multiple themes
    board = chess.Board("r2qkb1r/ppp2ppp/2n1bn2/2BpP3/3P4/3B1N2/PPP2PPP/RNBQK2R w KQkq - 0 8")
    
    print(f"\nPosition: {board.fen()}")
    print("Themes: King in center, pieces developed, tactical opportunities")
    
    moves = list(board.legal_moves)
    print(f"\nTotal legal moves: {len(moves)}")
    
    # Current ordering
    from v7p3r import SearchOptions
    options = SearchOptions()
    ordered_moves = engine._order_moves_enhanced(board, moves.copy(), 0, options)
    
    print(f"\n🔄 Current V8.2 Move Ordering (Top 10):")
    for i, move in enumerate(ordered_moves[:10], 1):
        move_type = []
        if board.is_capture(move):
            move_type.append("Capture")
        if board.is_check():
            board.push(move)
            if board.is_check():
                move_type.append("Check")
            board.pop()
        else:
            board.push(move)
            if board.is_check():
                move_type.append("Check")
            board.pop()
        if move.promotion:
            move_type.append("Promotion")
        if not move_type:
            move_type.append("Quiet")
        
        print(f"  {i:2d}. {move} ({', '.join(move_type)})")
    
    # Evaluate position
    eval_score = engine._evaluate_position_deterministic(board)
    print(f"\nPosition evaluation: {eval_score:+.2f}")
    
    # Show efficiency improvements
    print(f"\nV8.2 Efficiency Improvements:")
    print(f"- Total legal moves: {len(moves)}")
    print(f"- Moves after pruning: {len(ordered_moves)}")
    print(f"- Pruning efficiency: {((len(moves) - len(ordered_moves)) / len(moves) * 100):.1f}% reduction")

def main():
    """Run complete move ordering analysis for V8.2"""
    print("V7P3R MOVE ORDERING vs EVALUATION ALIGNMENT ANALYSIS")
    print("Analyzing current V8.2 system and proposed improvements")
    print("=" * 70)
    
    # User's proposed priorities
    proposed_priorities = [
        "Mates",
        "Captures (us on them)",
        "Threats (them against us) - defensive moves",
        "Attacks on multiple enemy pieces",
        "Attacks on enemy piece with lower value piece",
        "Defense of our own pieces",
        "Attacks on squares surrounding opponent king",
        "Moves to squares surrounding our own king",
        "Promotions and castling (slight bump)",
        "Truncate remaining moves later in PV"
    ]
    
    analyze_current_move_ordering()
    analyze_current_evaluation()
    analyze_alignment_issues()
    analyze_proposed_ordering(proposed_priorities)
    test_alignment_with_position()
    
    print("\n🎯 CONCLUSION & RECOMMENDATIONS")
    print("=" * 60)
    
    print("\n✅ STRENGTHS OF PROPOSED APPROACH:")
    print("- Excellent alignment with evaluation priorities")
    print("- Addresses critical gaps (threats, defense, king safety)")
    print("- Maintains tactical sharpness while adding positional awareness")
    print("- Strategic move truncation could improve efficiency")
    print("- V8.2 already includes mate detection as highest priority")
    
    print("\n🚧 IMPLEMENTATION CONSIDERATIONS:")
    print("- Need efficient threat detection algorithms")
    print("- Multi-piece attack detection requires careful implementation")
    print("- King safety zones need to be well-defined")
    print("- Move truncation needs careful tuning to avoid missing tactics")
    print("- V8.2 foundation is solid - ready for enhancements")
    
    print("\n🎖️  EXPECTED BENEFITS:")
    print("- Better alpha-beta pruning efficiency")
    print("- More consistent tactical/positional balance")
    print("- Improved time management through smart move pruning")
    print("- Enhanced alignment between search and evaluation")
    print("- Building on stable V8.1 deterministic foundation")

if __name__ == "__main__":
    main()
