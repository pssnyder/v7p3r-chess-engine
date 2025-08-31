#!/usr/bin/env python3
"""
V10 Tactical Enhancement Integration Script
Swaps in the new tactical enhanced scoring calculation
"""

import shutil
import os
from datetime import datetime

def backup_and_integrate_tactical_enhancement():
    """Backup current scoring and integrate tactical enhancement"""
    
    print("🚀 V10 TACTICAL ENHANCEMENT INTEGRATION")
    print("=" * 50)
    
    # Paths
    src_dir = "src"
    current_scoring = os.path.join(src_dir, "v7p3r_scoring_calculation.py")
    tactical_enhanced = os.path.join(src_dir, "v7p3r_scoring_calculation_tactical_enhanced.py")
    main_engine = os.path.join(src_dir, "v7p3r.py")
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backups/v10_tactical_enhancement_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    print(f"📁 Creating backup in: {backup_dir}")
    shutil.copy2(current_scoring, os.path.join(backup_dir, "v7p3r_scoring_calculation_original.py"))
    shutil.copy2(main_engine, os.path.join(backup_dir, "v7p3r_original.py"))
    
    # Read current main engine file
    with open(main_engine, 'r', encoding='utf-8') as f:
        engine_content = f.read()
    
    print("🔧 Updating import statement in main engine...")
    
    # Update import statement
    old_import = "from v7p3r_scoring_calculation import V7P3RScoringCalculationClean"
    new_import = "from v7p3r_scoring_calculation_tactical_enhanced import V7P3RScoringCalculationTacticalEnhanced"
    
    # Update class instantiation
    old_class = "V7P3RScoringCalculationClean"
    new_class = "V7P3RScoringCalculationTacticalEnhanced"
    
    # Replace imports and class references
    updated_content = engine_content.replace(old_import, new_import)
    updated_content = updated_content.replace(old_class, new_class)
    
    # Write updated engine file
    with open(main_engine, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("✅ Integration complete!")
    print(f"📋 CHANGES MADE:")
    print(f"   • Backed up original files to {backup_dir}")
    print(f"   • Updated {main_engine} to use tactical enhanced scoring")
    print(f"   • New tactical patterns available:")
    print(f"     - Pin detection and exploitation")
    print(f"     - Fork detection (all pieces)")
    print(f"     - Skewer detection")
    print(f"     - Discovered attack detection")
    print(f"     - Deflection opportunities")
    print(f"     - Removing the guard tactics")
    print(f"     - Double check detection")
    print(f"     - Battery formation")
    print(f"     - Piece defense coordination")
    print(f"     - Enhanced endgame (edge-driving)")
    
    print(f"\n🎯 V10 now has comprehensive tactical awareness!")
    print(f"🏁 Ready for tactical testing and validation")
    
    return backup_dir

def restore_from_backup(backup_dir):
    """Restore original files from backup"""
    print(f"🔄 RESTORING FROM BACKUP: {backup_dir}")
    
    src_dir = "src"
    current_scoring = os.path.join(src_dir, "v7p3r_scoring_calculation.py")
    main_engine = os.path.join(src_dir, "v7p3r.py")
    
    # Restore files
    shutil.copy2(os.path.join(backup_dir, "v7p3r_scoring_calculation_original.py"), current_scoring)
    shutil.copy2(os.path.join(backup_dir, "v7p3r_original.py"), main_engine)
    
    print("✅ Restore complete!")

if __name__ == "__main__":
    backup_dir = backup_and_integrate_tactical_enhancement()
    
    print(f"\n💡 TO RESTORE ORIGINAL V10:")
    print(f"python -c \"from v10_tactical_enhancement import restore_from_backup; restore_from_backup('{backup_dir}')\"")
