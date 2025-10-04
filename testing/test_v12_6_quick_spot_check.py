#!/usr/bin/env python3
"""
Quick V12.6 vs V12.2 Spot Check
Fast analysis of key differences
"""

import subprocess
import time
from pathlib import Path


def quick_engine_test(engine_path, name):
    """Quick test of engine on starting position"""
    print(f"\n--- {name} Quick Test ---")
    
    try:
        # Very simple test - just get a move from starting position
        result = subprocess.run(
            [str(engine_path)],
            input="uci\nposition startpos\ngo movetime 500\nquit\n",
            text=True,
            capture_output=True,
            timeout=3
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            
            # Extract engine info
            engine_id = "Unknown"
            bestmove = "none"
            last_info = ""
            
            for line in lines:
                if line.startswith("id name"):
                    engine_id = line.replace("id name ", "")
                elif line.startswith("bestmove"):
                    bestmove = line.split()[1] if len(line.split()) > 1 else "none"
                elif line.startswith("info") and "depth" in line:
                    last_info = line
            
            print(f"✅ ID: {engine_id}")
            print(f"✅ Move: {bestmove}")
            if last_info:
                print(f"✅ Info: {last_info}")
            
            return {
                "id": engine_id,
                "move": bestmove,
                "info": last_info,
                "success": True
            }
        else:
            print(f"❌ Failed: {result.stderr}")
            return {"success": False}
            
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout")
        return {"success": False}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False}


def main():
    base_path = Path("s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine")
    v12_2_path = base_path / "dist" / "V7P3R_v12.2.exe"
    v12_6_path = base_path / "dist" / "V7P3R_v12.6.exe"
    
    print("V7P3R V12.6 vs V12.2 QUICK SPOT CHECK")
    print("="*45)
    
    # Verify engines exist
    if not v12_2_path.exists():
        print(f"❌ V12.2 not found: {v12_2_path}")
        return
    if not v12_6_path.exists():
        print(f"❌ V12.6 not found: {v12_6_path}")
        return
        
    print(f"Engines found, testing...")
    
    # Quick tests
    v12_2_result = quick_engine_test(v12_2_path, "V12.2")
    v12_6_result = quick_engine_test(v12_6_path, "V12.6")
    
    # Summary
    print("\n" + "="*45)
    print("QUICK COMPARISON SUMMARY")
    print("="*45)
    
    if v12_2_result["success"] and v12_6_result["success"]:
        print(f"V12.2 Opening Move: {v12_2_result['move']}")
        print(f"V12.6 Opening Move: {v12_6_result['move']}")
        
        if v12_2_result['move'] == v12_6_result['move']:
            print("✅ SAME opening move - engines agree")
            status = "CONSISTENT"
        else:
            print("⚠️  DIFFERENT opening moves - investigating...")
            status = "DIFFERENT" 
        
        print(f"\nV12.2 Last Info: {v12_2_result['info']}")
        print(f"V12.6 Last Info: {v12_6_result['info']}")
        
    elif v12_2_result["success"]:
        print("❌ V12.6 FAILED - V12.2 working")
        status = "V12.6_BROKEN"
    elif v12_6_result["success"]:
        print("❌ V12.2 FAILED - V12.6 working")
        status = "V12.2_BROKEN"
    else:
        print("❌ BOTH ENGINES FAILED")
        status = "BOTH_BROKEN"
    
    print(f"\n🎯 Status: {status}")
    
    if status == "DIFFERENT":
        print("\n🔍 NEXT STEPS:")
        print("1. Investigate evaluation differences between engines")
        print("2. Check if V12.6 nudge removal affected opening book")
        print("3. Run deeper analysis to find root cause")
        print("4. Consider V12.6 may need evaluation tuning")
    elif status == "CONSISTENT":
        print("\n✅ GOOD NEWS:")
        print("1. Both engines working and agree on starting position")
        print("2. Ready for deeper arena tournament testing")
        print("3. V12.6 nudge removal appears successful")
    
    return status


if __name__ == "__main__":
    main()