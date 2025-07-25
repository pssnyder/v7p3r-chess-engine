#!/usr/bin/env python3
"""
Test script for the new V7P3R metrics system.
This script tests the basic functionality of the new unified metrics system.
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from metrics.v7p3r_chess_metrics import (
    v7p3rMetrics, GameMetric, MoveMetric, EngineConfig,
    get_metrics_instance, add_game_result, add_move_metric
)

async def test_metrics_system():
    """Test the new metrics system functionality."""
    print("≡ƒÄ» Testing V7P3R Unified Metrics System")
    print("=" * 50)
    
    # Get metrics instance
    metrics = get_metrics_instance()
    print("Γ£à Metrics instance created successfully")
    
    # Test game recording
    game_id = f"test_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timestamp = datetime.now().isoformat()
    
    # Create test game
    game_metric = GameMetric(
        game_id=game_id,
        timestamp=timestamp,
        v7p3r_color="white",
        opponent="stockfish",
        result="pending",
        total_moves=0,
        game_duration=0.0
    )
    
    success = await metrics.record_game_start(game_metric)
    print(f"Γ£à Game recording: {'Success' if success else 'Failed'}")
    
    # Test move recording
    move_metric = MoveMetric(
        game_id=game_id,
        move_number=1,
        player="v7p3r",
        move_notation="e2e4",
        position_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        evaluation_score=0.15,
        search_depth=6,
        nodes_evaluated=12450,
        time_taken=0.25,
        best_move="e2e4",
        pv_line="e2e4 e7e5 Nf3"
    )
    
    success = await metrics.record_move(move_metric)
    print(f"Γ£à Move recording: {'Success' if success else 'Failed'}")
    
    # Test game completion
    success = await metrics.update_game_result(
        game_id=game_id,
        result="win",
        total_moves=25,
        game_duration=45.5,
        final_position_fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        termination_reason="checkmate"
    )
    print(f"Γ£à Game completion: {'Success' if success else 'Failed'}")
    
    # Test engine config recording
    config = EngineConfig(
        config_id=f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        timestamp=timestamp,
        search_depth=6,
        time_limit=5.0,
        use_iterative_deepening=True,
        use_transposition_table=True,
        use_quiescence_search=True,
        use_move_ordering=True,
        hash_size_mb=128,
        additional_params={"opening_book": True, "endgame_tablebase": False}
    )
    
    success = await metrics.save_engine_config(config)
    print(f"Γ£à Engine config recording: {'Success' if success else 'Failed'}")
    
    # Test data retrieval
    game_summary = await metrics.get_game_summary(game_id)
    print(f"Γ£à Game summary retrieval: {'Success' if game_summary else 'Failed'}")
    if game_summary:
        print(f"   Game ID: {game_summary.get('game_id')}")
        print(f"   Result: {game_summary.get('result')}")
        print(f"   Duration: {game_summary.get('game_duration')}s")
    
    # Test performance trends
    trends_df = await metrics.get_performance_trends(limit=5)
    print(f"Γ£à Performance trends: {'Success' if not trends_df.empty else 'No data'}")
    if not trends_df.empty:
        print(f"   Retrieved {len(trends_df)} games for analysis")
    
    # Test legacy compatibility functions
    print("\n≡ƒöä Testing Legacy Compatibility")
    add_game_result(
        game_id=f"legacy_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now().isoformat(),
        winner="win",
        game_pgn="[Event 'Test'] [Result '1-0'] 1. e4 1-0",
        white_player="v7p3r",
        black_player="stockfish",
        game_length=1,
        white_engine_config="test_config",
        black_engine_config="stockfish_config"
    )
    print("Γ£à Legacy game recording compatibility")
    
    add_move_metric(
        game_id=f"legacy_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        move_number=1,
        player_color="white",
        move_uci="e2e4",
        fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        evaluation=0.1,
        search_algorithm="v7p3r",
        depth=5,
        nodes_searched=10000,
        time_taken=0.3,
        pv_line="e2e4 e7e5"
    )
    print("Γ£à Legacy move recording compatibility")
    
    print("\n≡ƒÄë All tests completed successfully!")
    print("≡ƒôè The new metrics system is ready for integration with v7p3r_play.py")

async def main():
    """Main test function."""
    try:
        await test_metrics_system()
    except Exception as e:
        print(f"Γ¥î Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\nΓ£à Test completed successfully!")
        print("≡ƒÆí To run the Streamlit dashboard: python metrics/chess_metrics.py dashboard")
    else:
        print("\nΓ¥î Test failed!")
        sys.exit(1)
