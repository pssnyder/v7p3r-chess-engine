#!/usr/bin/env python3

"""
V7P3R Move Ordering Improvement Patch

Based on move ordering analysis, implement targeted improvements:
1. Central pawn opening bonuses (e2e4, d2d4)
2. Increased check move prioritization  
3. Knight development improvements
"""

import chess
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def apply_move_ordering_improvements():
    """Apply move ordering improvements to V7P3R"""
    
    v7p3r_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'v7p3r.py')
    
    if not os.path.exists(v7p3r_path):
        print(f"❌ V7P3R engine not found: {v7p3r_path}")
        return
    
    print("🔧 Applying Move Ordering Improvements to V7P3R")
    print("=" * 50)
    
    # Read current file
    with open(v7p3r_path, 'r') as f:
        content = f.read()
    
    # Define the improvement patches
    improvements = [
        {
            "name": "Central Pawn Opening Bonus",
            "description": "Add bonus for e2e4, d2d4 central pawn moves in opening",
            "find": "            # 3. Checks (high priority for tactical play)",
            "replace": """            # 2.5. Opening central pawn moves (NEW - V14.3 improvement)
            if piece and piece.piece_type == chess.PAWN and self._is_opening_position(board):
                central_pawn_moves = ['e2e4', 'd2d4', 'e7e5', 'd7d5']
                if move.uci() in central_pawn_moves:
                    pawn_advances.append((80.0, move))  # High priority for central pawns
                    continue
            
            # 3. Checks (INCREASED priority for tactical play)"""
        },
        {
            "name": "Increased Check Priority",
            "description": "Increase check move scoring for better tactical play", 
            "find": "                checks.append((tactical_bonus, move))",
            "replace": "                checks.append((tactical_bonus + 60.0, move))  # Increased check bonus"
        },
        {
            "name": "Knight Development Improvement",
            "description": "Penalize knight moves to edge squares in opening",
            "find": "                # Development moves (knights, bishops moving from starting squares)",
            "replace": """                # Development moves (knights, bishops moving from starting squares)
                # NEW: Penalize knight moves to edge in opening
                if (piece.piece_type == chess.KNIGHT and self._is_opening_position(board) and 
                    move.from_square in [chess.B1, chess.G1, chess.B8, chess.G8] and
                    move.to_square in [chess.A3, chess.H3]):
                    development.append((10.0, move))  # Low priority for edge knights
                    continue"""
        },
        {
            "name": "Opening Position Detection Method",
            "description": "Add helper method to detect opening positions",
            "find": "    def get_performance_report(self) -> str:",
            "replace": """    def _is_opening_position(self, board: chess.Board) -> bool:
        \"\"\"V14.3: Detect if we're still in the opening phase\"\"\"
        # Simple heuristic: opening if most pieces are still on starting squares
        piece_count = len(board.piece_map())
        return piece_count >= 28  # Most pieces still on board
    
    def get_performance_report(self) -> str:"""
        }
    ]
    
    # Apply each improvement
    modified_content = content
    applied_count = 0
    
    for improvement in improvements:
        if improvement["find"] in modified_content:
            print(f"✅ Applying: {improvement['name']}")
            print(f"   {improvement['description']}")
            modified_content = modified_content.replace(improvement["find"], improvement["replace"])
            applied_count += 1
        else:
            print(f"❌ Could not apply: {improvement['name']}")
            print(f"   Pattern not found: {improvement['find'][:50]}...")
    
    if applied_count > 0:
        # Create backup
        backup_path = v7p3r_path + '.backup_before_move_ordering_improvements'
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"\n💾 Backup created: {backup_path}")
        
        # Write improved version
        with open(v7p3r_path, 'w') as f:
            f.write(modified_content)
        
        print(f"\n🎯 Applied {applied_count}/{len(improvements)} improvements to V7P3R")
        print("   Improvements:")
        print("   • Central pawn opening bonuses (e2e4, d2d4)")
        print("   • Increased check move prioritization (+60 bonus)")
        print("   • Knight development penalties for edge moves")
        print("   • Opening position detection helper")
        
        print(f"\n🔍 Run move ordering test again to measure improvements:")
        print(f"   python testing/move_ordering_test.py")
        
    else:
        print("❌ No improvements could be applied - code structure may have changed")

if __name__ == "__main__":
    apply_move_ordering_improvements()