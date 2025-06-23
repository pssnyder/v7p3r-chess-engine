#!/usr/bin/env python3
"""
Database Diagnostics for V7P3R Chess Engine Metrics Store

This script provides diagnostic functions to check the state of the metrics database,
investigate empty dashboard issues, and troubleshoot data collection problems.

Author: V7P3R Testing Suite
Date: 2025-06-22
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from metrics.metrics_store import MetricsStore


def check_database_status(db_path="metrics/chess_metrics.db"):
    """
    Check the current status of the metrics database.
    
    Returns detailed information about:
    - Database existence
    - Table structure
    - Record counts
    - Data availability for dashboard
    """
    print("=== V7P3R Chess Metrics Database Diagnostics ===\n")
    
    # Check if database file exists
    if not os.path.exists(db_path):
        print(f"❌ Database file does not exist at: {db_path}")
        return False
    
    print(f"✅ Database file exists at: {db_path}")
    
    try:
        # Initialize metrics store
        store = MetricsStore(db_path=db_path)
        conn = store._get_connection()
        cursor = conn.cursor()
        
        # Check available tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"📋 Available tables: {tables}")
        
        # Check each table's record count
        print("\n📊 Table Record Counts:")
        for table in tables:
            if table != 'sqlite_sequence':  # Skip system table
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  {table}: {count} records")
        
        # Detailed analysis for key tables
        print("\n🔍 Detailed Analysis:")
        
        # Game Results Analysis
        if 'game_results' in tables:
            cursor.execute("SELECT COUNT(*) FROM game_results")
            game_count = cursor.fetchone()[0]
            
            if game_count > 0:
                cursor.execute("SELECT winner, COUNT(*) FROM game_results GROUP BY winner")
                results = cursor.fetchall()
                print(f"  Game Results: {game_count} total games")
                for winner, count in results:
                    print(f"    {winner}: {count} games")
                
                # Check for engine type data
                try:
                    cursor.execute("SELECT DISTINCT white_engine_type, black_engine_type FROM game_results WHERE white_engine_type IS NOT NULL")
                    engine_combos = cursor.fetchall()
                    if engine_combos:
                        print(f"  Engine combinations found: {len(engine_combos)}")
                        for white, black in engine_combos[:5]:  # Show first 5
                            print(f"    White: {white} vs Black: {black}")
                    else:
                        print("  ⚠️  No engine type data found in game_results")
                except Exception as e:
                    print(f"  ⚠️  Error checking engine types: {e}")
            else:
                print("  ❌ No game results found - this explains empty dashboard!")
        
        # Move Metrics Analysis
        if 'move_metrics' in tables:
            cursor.execute("SELECT COUNT(*) FROM move_metrics")
            move_count = cursor.fetchone()[0]
            
            if move_count > 0:
                cursor.execute("SELECT DISTINCT player_color FROM move_metrics")
                colors = [row[0] for row in cursor.fetchall()]
                print(f"  Move Metrics: {move_count} total moves")
                print(f"    Player colors: {colors}")
                
                # Check for evaluation data
                cursor.execute("SELECT COUNT(*) FROM move_metrics WHERE evaluation IS NOT NULL")
                eval_count = cursor.fetchone()[0]
                print(f"    Moves with evaluation: {eval_count}")
                
                # Check for engine type in move metrics
                cursor.execute("SELECT DISTINCT engine_type FROM move_metrics WHERE engine_type IS NOT NULL")
                move_engines = [row[0] for row in cursor.fetchall()]
                if move_engines:
                    print(f"    Engine types in moves: {move_engines}")
                else:
                    print("  ⚠️  No engine type data in move_metrics")
            else:
                print("  ❌ No move metrics found")
        
        # Config Settings Analysis
        if 'config_settings' in tables:
            cursor.execute("SELECT COUNT(*) FROM config_settings")
            config_count = cursor.fetchone()[0]
            
            if config_count > 0:
                print(f"  Config Settings: {config_count} configurations")
                try:
                    cursor.execute("SELECT DISTINCT white_engine_type, black_engine_type FROM config_settings WHERE white_engine_type IS NOT NULL")
                    config_engines = cursor.fetchall()
                    if config_engines:
                        print(f"    Engine configurations: {len(config_engines)}")
                    else:
                        print("  ⚠️  No engine type data in config_settings")
                except Exception as e:
                    print(f"  ⚠️  Error checking config engine types: {e}")
            else:
                print("  ❌ No config settings found")
        
        # Check for log_entries table (might be missing)
        if 'log_entries' in tables:
            cursor.execute("SELECT COUNT(*) FROM log_entries")
            log_count = cursor.fetchone()[0]
            print(f"  Log Entries: {log_count} entries")
        else:
            print("  ⚠️  log_entries table does not exist - log parsing not implemented")
        
        # Check for metrics table (computed metrics)
        if 'metrics' in tables:
            cursor.execute("SELECT COUNT(*) FROM metrics")
            metrics_count = cursor.fetchone()[0]
            print(f"  Computed Metrics: {metrics_count} metrics")
            
            if metrics_count > 0:
                cursor.execute("SELECT DISTINCT metric_name FROM metrics LIMIT 10")
                metric_names = [row[0] for row in cursor.fetchall()]
                print(f"    Available metrics: {metric_names}")
        else:
            print("  ⚠️  metrics table does not exist - no computed metrics available")
        
        store.close()
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing database: {e}")
        return False


def diagnose_empty_dashboard():
    """
    Specifically diagnose why the chess metrics dashboard might be empty.
    """
    print("\n=== Dashboard Empty Issue Diagnosis ===\n")
    
    # Check if data collection has been run
    print("🔍 Checking data collection status...")
    
    try:
        store = MetricsStore()
        
        # Try to run data collection manually
        print("📥 Running data collection manually...")
        store.collect_all_data()
        
        # Check if this populated any data
        conn = store._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM game_results")
        game_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM move_metrics")
        move_count = cursor.fetchone()[0]
        
        print(f"After data collection attempt:")
        print(f"  Games: {game_count}")
        print(f"  Moves: {move_count}")
        
        if game_count == 0:
            print("\n❌ ISSUE FOUND: No game data collected!")
            print("Possible causes:")
            print("  1. No PGN/YAML files in games/ directory")
            print("  2. Files not matching expected pattern (eval_game_*.pgn/yaml)")
            print("  3. File parsing errors")
            
            # Check games directory
            games_dir = "games"
            if os.path.exists(games_dir):
                pgn_files = [f for f in os.listdir(games_dir) if f.endswith('.pgn')]
                yaml_files = [f for f in os.listdir(games_dir) if f.endswith('.yaml')]
                print(f"\nFound in games/ directory:")
                print(f"  PGN files: {len(pgn_files)}")
                print(f"  YAML files: {len(yaml_files)}")
                
                if pgn_files:
                    print(f"  Sample PGN files: {pgn_files[:3]}")
                if yaml_files:
                    print(f"  Sample YAML files: {yaml_files[:3]}")
            else:
                print(f"❌ Games directory '{games_dir}' does not exist!")
        
        store.close()
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")


def main():
    """Main diagnostic function."""
    print("Starting V7P3R Chess Metrics Database Diagnostics...\n")
    
    # Run database status check
    db_ok = check_database_status()
    
    if db_ok:
        # Run dashboard-specific diagnosis
        diagnose_empty_dashboard()
        
        print("\n=== Recommendations ===")
        print("1. If no game data: Run chess games to generate PGN/YAML files")
        print("2. If files exist but not imported: Check file naming patterns")
        print("3. If data exists but dashboard empty: Check dashboard queries")
        print("4. Consider running store.collect_all_data() manually")
    else:
        print("\n❌ Database issues prevent further diagnosis")
    
    print("\nDiagnostics complete!")


if __name__ == "__main__":
    main()
