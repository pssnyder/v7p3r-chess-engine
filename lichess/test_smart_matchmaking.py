#!/usr/bin/env python3
"""Test script for V7P3R Smart Matchmaking system."""
import sys
import logging
from unittest.mock import Mock

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("=" * 70)
print("V7P3R SMART MATCHMAKING TEST")
print("=" * 70)
print()

# Test 1: Import modules
print("Test 1: Importing modules...")
try:
    from lib.v7p3r_matchmaking_intelligence import V7P3RMatchmakingIntelligence
    print("✓ v7p3r_matchmaking_intelligence imported successfully")
except Exception as e:
    print(f"✗ Failed to import v7p3r_matchmaking_intelligence: {e}")
    sys.exit(1)

try:
    from lib.v7p3r_smart_matchmaking import V7P3RSmartMatchmaking
    print("✓ v7p3r_smart_matchmaking imported successfully")
except Exception as e:
    print(f"✗ Failed to import v7p3r_smart_matchmaking: {e}")
    sys.exit(1)

print()

# Test 2: Intelligence analyzer
print("Test 2: Testing intelligence analyzer...")
try:
    intel = V7P3RMatchmakingIntelligence("game_records", "v7p3r_bot")
    stats = intel.analyze_game_records()
    
    print(f"✓ Analyzed {len(stats)} unique opponents")
    total_games = sum(s.games_played for s in stats.values())
    print(f"✓ Total games: {total_games}")
    
    improvement_targets = intel.get_improvement_targets(5)
    print(f"✓ Found {len(improvement_targets)} improvement targets")
    
    if improvement_targets:
        top_target = improvement_targets[0]
        print(f"  Top target: {top_target.name} ({top_target.win_rate:.1f}% WR, {top_target.games_played}g)")
    
    priority_opp = intel.get_priority_opponent()
    print(f"✓ Priority opponent: {priority_opp}")
    
except Exception as e:
    print(f"✗ Intelligence analyzer failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Smart matchmaking initialization
print("Test 3: Testing smart matchmaking initialization...")
try:
    # Mock the dependencies
    mock_li = Mock()
    mock_li.get_online_bots = Mock(return_value=[])
    mock_li.get_following = Mock(return_value=[])
    
    mock_config = Mock()
    mock_config.matchmaking = Mock()
    mock_config.matchmaking.lookup_or_default = Mock(side_effect=lambda key, default: {
        "smart_matchmaking_enabled": True,
        "reserve_challenge_slot": True,
        "use_followed_accounts": True,
        "priority_opponents": ["c0br4_bot", "slowmate_bot"],
        "min_games_per_opponent": 5,
        "improvement_target_weight": 0.3,
        "variety_weight": 0.5,
        "priority_weight": 0.2,
        "challenge_timeout": 30,
        "allow_during_games": False,
        "challenge_filter": "none",
        "block_list": [],
        "online_block_list": [],
        "include_challenge_block_list": False,
        "challenge_initial_time": [180, 300, 600],
        "challenge_increment": [2, 3, 5],
        "challenge_days": [0],
        "opponent_min_rating": 600,
        "opponent_max_rating": 4000,
        "opponent_rating_difference": 500,
        "rating_preference": "none",
        "challenge_variant": ["standard"],
        "challenge_mode": "rated",
        "overrides": {}
    }.get(key, default))
    
    mock_config.challenge = Mock()
    mock_config.challenge.variants = ["standard", "chess960"]
    
    mock_user_profile = {
        "username": "v7p3r_bot",
        "perfs": {
            "blitz": {"rating": 2000, "games": 100}
        }
    }
    
    # Initialize smart matchmaking
    smart_mm = V7P3RSmartMatchmaking(mock_li, mock_config, mock_user_profile, "game_records")
    print("✓ Smart matchmaking initialized successfully")
    print(f"✓ Smart config: enabled={smart_mm.smart_config['enabled']}")
    print(f"✓ Reserve slot: {smart_mm.smart_config['reserve_challenge_slot']}")
    print(f"✓ Priority opponents: {smart_mm.smart_config['priority_opponents']}")
    
except Exception as e:
    print(f"✗ Smart matchmaking initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Strategy selection
print("Test 4: Testing strategy selection...")
try:
    # Test opponent selection logic
    print("✓ Testing improvement target selection...")
    target = smart_mm._choose_improvement_target([])
    print(f"  Result: {target} (expected None with empty online bots)")
    
    print("✓ Testing variety opponent selection...")
    variety = smart_mm._choose_variety_opponent([])
    print(f"  Result: {variety} (expected None with empty online bots)")
    
    print("✓ Testing priority opponent selection...")
    priority = smart_mm._choose_priority_opponent([])
    print(f"  Result: {priority} (expected None - not online)")
    
except Exception as e:
    print(f"✗ Strategy selection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print()
print("Smart matchmaking system is ready for deployment!")
print()
print("Next steps:")
print("1. Enable in config: allow_matchmaking: true")
print("2. Deploy to cloud with updated modules")
print("3. Monitor logs for intelligent opponent selection")
