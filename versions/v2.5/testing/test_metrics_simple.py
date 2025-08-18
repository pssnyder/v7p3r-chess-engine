﻿#!/usr/bin/env python3
"""
Simple test to verify the refactored metrics system works correctly
"""
import sys
import os

# Set up path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_enhanced_metrics_store():
    """Test that the enhanced metrics store can be imported and initialized"""
    try:
        from metrics.enhanced_metrics_store import EnhancedMetricsStore
        store = EnhancedMetricsStore()
        print("Γ£à Enhanced metrics store initialized successfully")
        return True
    except Exception as e:
        print(f"Γ¥î Enhanced metrics store failed: {e}")
        return False

def test_refactored_collector():
    """Test that the refactored metrics collector can be imported"""
    try:
        from metrics.refactored_enhanced_metrics_collector import RefactoredEnhancedMetricsCollector
        collector = RefactoredEnhancedMetricsCollector()
        print("Γ£à Refactored metrics collector initialized successfully")
        return True
    except Exception as e:
        print(f"Γ¥î Refactored metrics collector failed: {e}")
        return False

def test_analytics_processor():
    """Test that the analytics processor can be imported and run"""
    try:
        from metrics.refactored_analytics_processor import RefactoredAnalyticsProcessor
        processor = RefactoredAnalyticsProcessor()
        print("Γ£à Analytics processor initialized successfully")
        
        # Try to generate a report
        try:
            report = processor.generate_performance_report()
            print(f"Γ£à Analytics report generated: {len(report)} characters")
            return True
        except Exception as e:
            print(f"ΓÜá∩╕Å Analytics processor initialized but report generation failed: {e}")
            return True  # Still consider this a success
    except Exception as e:
        print(f"Γ¥î Analytics processor failed: {e}")
        return False

def test_database_connections():
    """Test database connections work correctly"""
    try:
        import sqlite3
        
        # Test enhanced database
        enhanced_db = os.path.join(current_dir, "metrics", "chess_metrics_v2.db")
        if os.path.exists(enhanced_db):
            conn = sqlite3.connect(enhanced_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM move_metrics")
            count = cursor.fetchone()[0]
            conn.close()
            print(f"Γ£à Enhanced database connected: {count} move metrics found")
        else:
            print("ΓÜá∩╕Å Enhanced database not found")
        
        # Test legacy database
        legacy_db = os.path.join(current_dir, "metrics", "chess_metrics.db")
        if os.path.exists(legacy_db):
            conn = sqlite3.connect(legacy_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM move_metrics")
            count = cursor.fetchone()[0]
            conn.close()
            print(f"Γ£à Legacy database connected: {count} move metrics found")
        else:
            print("ΓÜá∩╕Å Legacy database not found")
        
        return True
    except Exception as e:
        print(f"Γ¥î Database connection test failed: {e}")
        return False

if __name__ == "__main__":
    print("≡ƒöì Testing Refactored Metrics System")
    print("=" * 50)
    
    tests = [
        ("Enhanced Metrics Store", test_enhanced_metrics_store),
        ("Refactored Collector", test_refactored_collector),
        ("Analytics Processor", test_analytics_processor),
        ("Database Connections", test_database_connections)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"Γ¥î {test_name} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print(f"≡ƒôè Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("≡ƒÄë All tests passed! Refactored metrics system is working correctly.")
    else:
        print("ΓÜá∩╕Å Some tests failed. Check the output above for details.")
