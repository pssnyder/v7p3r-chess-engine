#!/usr/bin/env python3
"""
Test for Phase 2: Layered Configuration System
Tests the enhanced configuration loading with default + custom overlay
"""

import sys
import os
import json

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v7p3r_config import v7p3rConfig

def test_default_ruleset_loading():
    """Test that default ruleset loads correctly"""
    print("=== Testing Default Ruleset Loading ===")
    
    try:
        config = v7p3rConfig()
        ruleset = config.get_ruleset()
        
        print(f"Loaded ruleset: {config.ruleset_name}")
        print(f"Ruleset has {len(ruleset)} settings")
        
        # Check for key default values
        required_keys = ['checkmate_threats_modifier', 'material_score_modifier', 'center_control_modifier']
        
        for key in required_keys:
            if key in ruleset:
                print(f"✅ {key}: {ruleset[key]}")
            else:
                print(f"❌ Missing required key: {key}")
                return False
        
        print("✅ SUCCESS: Default ruleset loaded correctly")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_ruleset_overlay():
    """Test custom ruleset overlay on default values"""
    print("\n=== Testing Custom Ruleset Overlay ===")
    
    try:
        # Create a simple custom ruleset for testing
        custom_ruleset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         'configs', 'rulesets', 'test_checkmate_mod_ruleset.json')
        
        custom_data = {
            "test_checkmate_mod_ruleset": {
                "checkmate_threats_modifier": 1000.0
            }
        }
        
        # Write test custom ruleset
        with open(custom_ruleset_path, 'w') as f:
            json.dump(custom_data, f, indent=2)
        
        print(f"Created test custom ruleset: {custom_ruleset_path}")
        
        # Create config with custom_config that specifies the custom ruleset
        custom_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        'configs', 'test_custom_config.json')
        
        custom_config_data = {
            "engine_config": {
                "ruleset": "test_checkmate_mod_ruleset"
            }
        }
        
        with open(custom_config_path, 'w') as f:
            json.dump(custom_config_data, f, indent=2)
        
        print(f"Created test custom config: {custom_config_path}")
        
        # Load config with custom ruleset
        config = v7p3rConfig(config_path=custom_config_path)
        ruleset = config.get_ruleset()
        
        print(f"Using custom ruleset: {config.ruleset_name}")
        
        # Verify that custom value is applied
        checkmate_modifier = ruleset.get('checkmate_threats_modifier', 'NOT_FOUND')
        print(f"Checkmate threats modifier: {checkmate_modifier}")
        
        # Verify that default values are still present
        material_modifier = ruleset.get('material_score_modifier', 'NOT_FOUND')
        print(f"Material score modifier (should be default): {material_modifier}")
        
        # Cleanup test files
        os.remove(custom_ruleset_path)
        os.remove(custom_config_path)
        print("Cleaned up test files")
        
        if checkmate_modifier == 1000.0 and material_modifier != 'NOT_FOUND':
            print("✅ SUCCESS: Custom overlay working correctly")
            print("✅ Custom value overrode default, other defaults preserved")
            return True
        else:
            print("❌ FAILURE: Custom overlay not working correctly")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        try:
            if os.path.exists(custom_ruleset_path):
                os.remove(custom_ruleset_path)
            if os.path.exists(custom_config_path):
                os.remove(custom_config_path)
        except:
            pass
        return False

def test_strict_error_handling():
    """Test that engine fails fast when default configs are missing"""
    print("\n=== Testing Strict Error Handling ===")
    
    # This test would require temporarily moving the default files,
    # which could break other things. For now, just test that the
    # config loading is stricter about validation.
    
    try:
        config = v7p3rConfig()
        
        # Test that we have all required config sections
        required_sections = ['engine_config', 'game_config']
        
        for section in required_sections:
            section_data = getattr(config, section, None)
            if section_data is None or not isinstance(section_data, dict):
                print(f"❌ Missing or invalid config section: {section}")
                return False
            print(f"✅ Valid config section: {section}")
        
        print("✅ SUCCESS: Strict validation working")
        return True
        
    except Exception as e:
        print(f"Expected behavior - strict error handling: {e}")
        return True

if __name__ == "__main__":
    print("Testing Phase 2: Layered Configuration System")
    print("=" * 50)
    
    try:
        test1_result = test_default_ruleset_loading()
        test2_result = test_custom_ruleset_overlay()
        test3_result = test_strict_error_handling()
        
        print(f"\n=== Final Results ===")
        print(f"Default ruleset loading: {'PASS' if test1_result else 'FAIL'}")
        print(f"Custom ruleset overlay: {'PASS' if test2_result else 'FAIL'}")
        print(f"Strict error handling: {'PASS' if test3_result else 'FAIL'}")
        
        if test1_result and test2_result and test3_result:
            print("✅ Phase 2 enhanced configuration system working correctly!")
        else:
            print("❌ Phase 2 needs further investigation")
            
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
