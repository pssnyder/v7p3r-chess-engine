#!/usr/bin/env python3
"""
Database Migration Script for V7P3R Chess Engine Enhanced Metrics
This script backs up existing databases and sets up the new enhanced system
"""

import os
import shutil
import sqlite3
from datetime import datetime
import json

def backup_existing_databases():
    """
    Backup existing chess_metrics.db and chess_analytics.db
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"metrics/backup_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    backups_created = []
    
    # Backup chess_metrics.db
    metrics_path = "metrics/chess_metrics.db"
    if os.path.exists(metrics_path):
        backup_path = f"{backup_dir}/chess_metrics_backup.db"
        shutil.copy2(metrics_path, backup_path)
        backups_created.append(backup_path)
        print(f"✓ Backed up chess_metrics.db to {backup_path}")
    
    # Backup chess_analytics.db
    analytics_path = "metrics/chess_analytics.db"
    if os.path.exists(analytics_path):
        backup_path = f"{backup_dir}/chess_analytics_backup.db"
        shutil.copy2(analytics_path, backup_path)
        backups_created.append(backup_path)
        print(f"✓ Backed up chess_analytics.db to {backup_path}")
    
    return backups_created, backup_dir

def initialize_enhanced_database():
    """
    Initialize the new enhanced database structure
    """
    try:
        from enhanced_metrics_store import EnhancedMetricsStore
        
        # Initialize with new database name
        enhanced_store = EnhancedMetricsStore(db_path="metrics/chess_metrics_v2.db")
        print("✓ Enhanced database initialized successfully")
        
        return enhanced_store
        
    except Exception as e:
        print(f"✗ Error initializing enhanced database: {e}")
        return None

def migrate_legacy_data(enhanced_store, legacy_db_path):
    """
    Migrate data from legacy database to enhanced format
    """
    try:
        print(f"Migrating data from {legacy_db_path}...")
        enhanced_store.migrate_legacy_data(legacy_db_path)
        print("✓ Legacy data migration completed")
        
    except Exception as e:
        print(f"✗ Error during migration: {e}")

def create_migration_report(backup_dir, migration_success=True):
    """
    Create a report of the migration process
    """
    report = {
        "migration_timestamp": datetime.now().isoformat(),
        "backup_directory": backup_dir,
        "migration_success": migration_success,
        "enhanced_features": [
            "Detailed scoring breakdown (23+ evaluation components)",
            "Engine-specific metrics with proper attribution", 
            "Game phase and position analysis",
            "Search efficiency and performance metrics",
            "Enhanced database schema with optimized indexes",
            "Comprehensive position classification",
            "Material balance and piece activity tracking",
            "Opening book usage detection",
            "Tactical motif identification"
        ],
        "new_database_files": [
            "metrics/chess_metrics_v2.db"
        ],
        "backup_files": [],
        "migration_notes": "Enhanced metrics system provides comprehensive chess engine analysis capabilities"
    }
    
    report_path = f"{backup_dir}/migration_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Migration report created: {report_path}")
    return report_path

def test_enhanced_system():
    """
    Test the enhanced metrics system with a sample configuration
    """
    try:
        from enhanced_metrics_store import EnhancedMetricsStore
        from enhanced_scoring_collector import EnhancedScoringCollector
        
        print("Testing enhanced metrics system...")
        
        # Test database connection
        store = EnhancedMetricsStore(db_path="metrics/chess_metrics_v2.db")
        collector = EnhancedScoringCollector()
        
        # Test configuration storage
        test_config = {
            'name': 'v7p3r',
            'version': '2.0.0',
            'search_algorithm': 'minimax',
            'depth': 3,
            'max_depth': 5,
            'ruleset': 'enhanced_evaluation',
            'use_game_phase': True
        }
        
        config_id = store.add_engine_config(test_config)
        print(f"✓ Test configuration stored with ID: {config_id}")
        
        # Test game initialization
        store.start_game(
            game_id="test_game_001",
            white_player="v7p3r",
            black_player="stockfish",
            white_config=test_config,
            black_config={'name': 'stockfish', 'version': '16.0', 'search_algorithm': 'stockfish'},
            pgn_filename="test_game_001.pgn"
        )
        print("✓ Test game initialization successful")
        
        # Test move metric storage
        test_move_metric = {
            'game_id': 'test_game_001',
            'move_number': 1,
            'player_color': 'white',
            'move_san': 'e4',
            'move_uci': 'e2e4',
            'fen_before': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'fen_after': 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
            'search_algorithm': 'minimax',
            'depth_reached': 3,
            'nodes_searched': 1000,
            'time_taken': 0.1,
            'evaluation': 0.25,
            'game_phase': 'opening',
            'position_type': 'balanced',
            'material_balance': 0.0,
            'move_type': 'normal'
        }
        
        store.add_enhanced_move_metric(**test_move_metric)
        print("✓ Test move metric storage successful")
        
        # Test game completion
        store.finish_game(
            game_id="test_game_001",
            result="1-0",
            termination="checkmate",
            total_moves=25,
            game_duration=300.0
        )
        print("✓ Test game completion successful")
        
        print("✓ Enhanced metrics system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced metrics system test failed: {e}")
        return False

def main():
    """
    Main migration process
    """
    print("=" * 60)
    print("V7P3R CHESS ENGINE ENHANCED METRICS MIGRATION")
    print("=" * 60)
    print()
    
    print("Phase 1: Database Backup")
    print("-" * 30)
    backups, backup_dir = backup_existing_databases()
    print(f"Backups created in: {backup_dir}")
    print()
    
    print("Phase 2: Enhanced Database Initialization")
    print("-" * 30)
    enhanced_store = initialize_enhanced_database()
    if not enhanced_store:
        print("✗ Failed to initialize enhanced database. Exiting.")
        return False
    print()
    
    print("Phase 3: Legacy Data Migration")
    print("-" * 30)
    legacy_metrics_path = "metrics/chess_metrics.db"
    if os.path.exists(legacy_metrics_path):
        migrate_legacy_data(enhanced_store, legacy_metrics_path)
    else:
        print("No legacy chess_metrics.db found, skipping migration")
    print()
    
    print("Phase 4: System Testing")
    print("-" * 30)
    test_success = test_enhanced_system()
    print()
    
    print("Phase 5: Migration Report")
    print("-" * 30)
    report_path = create_migration_report(backup_dir, test_success)
    print()
    
    print("=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    
    if test_success:
        print("✓ Enhanced metrics system migration completed successfully!")
        print()
        print("New Features Available:")
        print("• Detailed scoring breakdown (23+ evaluation components)")
        print("• Engine-specific metrics with proper attribution")
        print("• Game phase and position analysis")
        print("• Search efficiency metrics")
        print("• Enhanced database performance")
        print("• Comprehensive position classification")
        print()
        print("Next Steps:")
        print("1. Run games using the updated v7p3r_play.py")
        print("2. Use the enhanced chess_metrics.py for analysis")
        print("3. Monitor the new database: metrics/chess_metrics_v2.db")
        print()
        print(f"Backup location: {backup_dir}")
        print(f"Migration report: {report_path}")
        
    else:
        print("✗ Migration completed with errors")
        print("Please check the error messages above and retry if needed")
        print(f"Your original databases are safely backed up in: {backup_dir}")
    
    return test_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
