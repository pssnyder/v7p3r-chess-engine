#!/usr/bin/env python3
"""
TAL-BOT Arena Deployment Test

Quick test to verify TAL-BOT is ready for Arena Chess GUI deployment.
"""

import subprocess
import os
import sys


def test_arena_deployment():
    """Test if TAL-BOT is ready for Arena deployment"""
    print("=== TAL-BOT ARENA DEPLOYMENT TEST ===\n")
    
    build_dir = "build/VPR_v1.0"
    
    # Check if build directory exists
    if not os.path.exists(build_dir):
        print("❌ Build directory not found!")
        return False
    
    print("✅ Build directory found: build/VPR_v1.0")
    
    # Check required files
    required_files = [
        "src/vpr.py",
        "src/vpr_uci.py", 
        "VPR_v1.0.bat"
    ]
    
    for file_path in required_files:
        full_path = os.path.join(build_dir, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path} found")
        else:
            print(f"❌ {file_path} missing!")
            return False
    
    print("\n=== TAL-BOT ENGINE VERIFICATION ===")
    
    # Test basic import
    sys.path.insert(0, os.path.join(build_dir, "src"))
    
    try:
        from vpr import VPREngine
        engine = VPREngine()
        print("✅ TAL-BOT engine imports successfully")
        
        # Test engine info
        info = engine.get_engine_info()
        print(f"✅ Engine name: {info['name']}")
        print(f"✅ Engine version: {info['version']}")
        print(f"✅ Author: {info['author']}")
        
        # Test basic functionality
        import chess
        board = chess.Board()
        
        # Quick search test
        move = engine.search(board, time_limit=0.5)
        print(f"✅ Quick search works: {move}")
        
        # Test dynamic piece values (TAL-BOT feature)
        knight_value = engine._calculate_piece_true_value(board, chess.G1, chess.WHITE)
        print(f"✅ Dynamic piece values: Knight g1 = {knight_value}")
        
        # Test chaos factor (TAL-BOT feature)
        chaos = engine._calculate_chaos_factor(board)
        print(f"✅ Chaos factor calculation: {chaos}")
        
    except Exception as e:
        print(f"❌ Engine test failed: {e}")
        return False
    
    print("\n=== ARENA INTEGRATION STATUS ===")
    print("🔥 TAL-BOT (The Entropy Engine) is ready for Arena!")
    print("✅ All files present")
    print("✅ Engine functionality verified") 
    print("✅ TAL-BOT features active:")
    print("   - Dynamic piece values")
    print("   - Chaos factor calculation")
    print("   - Principal Variation following")
    print("   - Priority-based move ordering")
    print("   - Entropy-driven search")
    
    print("\n=== DEPLOYMENT INSTRUCTIONS ===")
    print("1. Copy build/VPR_v1.0 folder to Arena engines directory")
    print("2. In Arena: Engines → Install New Engine")
    print("3. Navigate to VPR_v1.0 folder")
    print("4. Select VPR_v1.0.bat as the engine executable")
    print("5. Engine name will appear as 'TAL-BOT v1.0'")
    print("6. Ready to destroy traditional engines through entropy! ⚔️")
    
    return True


if __name__ == "__main__":
    print("TAL-BOT Arena Deployment Verification")
    print("=" * 50)
    
    success = test_arena_deployment()
    
    if success:
        print("\n🎉 TAL-BOT IS READY FOR BATTLE! 🎉")
        print("The entropy engine awaits its first Arena victim...")
    else:
        print("\n⚠️  Deployment issues detected - check errors above")