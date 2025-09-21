#!/usr/bin/env python3
"""
Enhanced Nudge System Validator
Tests the enhanced nudge system integration to ensure it works correctly
with tactical metadata, confidence scores, and improved move ordering.

Author: Pat Snyder
Created: September 21, 2025
"""

import sys
import os
import chess
import json
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine


class EnhancedNudgeValidator:
    """Validates the enhanced nudge system functionality"""
    
    def __init__(self):
        self.engine = V7P3REngine()
        self.test_results = {
            'database_loading': False,
            'enhanced_features': False,
            'tactical_bonuses': False,
            'confidence_scoring': False,
            'instant_moves': False,
            'move_ordering': False
        }
        
        print("🧪 Enhanced Nudge System Validator")
        print("=" * 40)
    
    def test_database_loading(self):
        """Test that enhanced database loads correctly"""
        print("📂 Testing enhanced database loading...")
        
        # Check if enhanced database was loaded
        db_size = len(self.engine.nudge_database)
        
        if db_size > 0:
            print(f"✅ Database loaded: {db_size} positions")
            
            # Sample a few positions to check enhanced format
            sample_count = 0
            enhanced_count = 0
            
            for pos_key, pos_data in list(self.engine.nudge_database.items())[:10]:
                sample_count += 1
                
                for move_key, move_data in pos_data.get('moves', {}).items():
                    if 'confidence' in move_data or 'tactical_info' in move_data:
                        enhanced_count += 1
                        break
            
            if enhanced_count > 0:
                print(f"✅ Enhanced features detected in {enhanced_count}/{sample_count} sampled positions")
                self.test_results['database_loading'] = True
                self.test_results['enhanced_features'] = True
            else:
                print("⚠️  No enhanced features detected - using original format")
                self.test_results['database_loading'] = True
        else:
            print("❌ No database loaded")
        
        print()
    
    def test_tactical_awareness(self):
        """Test tactical classification and bonuses"""
        print("⚔️  Testing tactical awareness...")
        
        tactical_positions_found = 0
        offensive_bonuses = 0
        defensive_bonuses = 0
        puzzle_bonuses = 0
        
        # Scan database for tactical positions
        for pos_key, pos_data in self.engine.nudge_database.items():
            for move_key, move_data in pos_data.get('moves', {}).items():
                tactical_info = move_data.get('tactical_info', {})
                
                if tactical_info:
                    tactical_positions_found += 1
                    classification = tactical_info.get('classification', 'development')
                    
                    if classification == 'offensive':
                        offensive_bonuses += 1
                    elif classification == 'defensive':
                        defensive_bonuses += 1
                    
                    if move_data.get('source') in ['puzzle', 'hybrid']:
                        puzzle_bonuses += 1
        
        if tactical_positions_found > 0:
            print(f"✅ Tactical positions found: {tactical_positions_found}")
            print(f"   Offensive: {offensive_bonuses}")
            print(f"   Defensive: {defensive_bonuses}")
            print(f"   Puzzle-derived: {puzzle_bonuses}")
            self.test_results['tactical_bonuses'] = True
        else:
            print("❌ No tactical metadata found")
        
        print()
    
    def test_confidence_scoring(self):
        """Test confidence score distribution"""
        print("🎯 Testing confidence scoring...")
        
        confidence_scores = []
        high_confidence_count = 0
        
        for pos_key, pos_data in self.engine.nudge_database.items():
            for move_key, move_data in pos_data.get('moves', {}).items():
                confidence = move_data.get('confidence')
                
                if confidence is not None:
                    confidence_scores.append(confidence)
                    if confidence >= 0.9:
                        high_confidence_count += 1
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            print(f"✅ Confidence scores found: {len(confidence_scores)} moves")
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   High confidence (≥0.9): {high_confidence_count}")
            self.test_results['confidence_scoring'] = True
        else:
            print("❌ No confidence scores found")
        
        print()
    
    def test_move_ordering_enhancement(self):
        """Test enhanced move ordering with tactical bonuses"""
        print("📊 Testing enhanced move ordering...")
        
        # Test with a position that should have nudge moves
        test_positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"  # Common opening
        ]
        
        nudge_moves_found = 0
        enhanced_bonuses = 0
        
        for test_fen in test_positions:
            try:
                board = chess.Board(test_fen)
                
                # Check for nudge moves in this position
                position_key = self.engine._get_position_key(board)
                
                if position_key in self.engine.nudge_database:
                    nudge_moves_found += 1
                    
                    # Test enhanced bonus calculation
                    legal_moves = list(board.legal_moves)
                    for move in legal_moves[:5]:  # Test first 5 moves
                        bonus = self.engine._get_nudge_bonus(board, move)
                        if bonus > 50:  # Enhanced bonuses should be higher
                            enhanced_bonuses += 1
                            break
                
            except Exception as e:
                print(f"   Error testing position {test_fen}: {e}")
        
        if nudge_moves_found > 0:
            print(f"✅ Nudge moves found in {nudge_moves_found}/{len(test_positions)} test positions")
            if enhanced_bonuses > 0:
                print(f"✅ Enhanced bonuses detected: {enhanced_bonuses}")
                self.test_results['move_ordering'] = True
        else:
            print("⚠️  No nudge moves found in test positions")
        
        print()
    
    def test_instant_move_detection(self):
        """Test enhanced instant move detection"""
        print("⚡ Testing instant move detection...")
        
        # Test with positions that might have high-confidence moves
        high_confidence_positions = 0
        instant_moves_detected = 0
        
        # Sample positions from database to find high-confidence moves
        for pos_key, pos_data in list(self.engine.nudge_database.items())[:20]:
            try:
                fen = pos_data.get('fen')
                if not fen:
                    continue
                
                board = chess.Board(fen)
                
                # Check for high-confidence moves
                has_high_confidence = False
                for move_key, move_data in pos_data.get('moves', {}).items():
                    confidence = move_data.get('confidence', 0.0)
                    if confidence >= 0.9:
                        has_high_confidence = True
                        break
                
                if has_high_confidence:
                    high_confidence_positions += 1
                    
                    # Test instant move detection
                    instant_move = self.engine._check_instant_nudge_move(board)
                    if instant_move:
                        instant_moves_detected += 1
                
            except Exception as e:
                continue
        
        if high_confidence_positions > 0:
            print(f"✅ High-confidence positions found: {high_confidence_positions}")
            print(f"   Instant moves detected: {instant_moves_detected}")
            if instant_moves_detected > 0:
                self.test_results['instant_moves'] = True
        else:
            print("⚠️  No high-confidence positions found in sample")
        
        print()
    
    def run_performance_test(self):
        """Quick performance test to ensure no major regression"""
        print("⏱️  Running performance test...")
        
        start_time = datetime.now()
        
        # Run a quick search on a standard position
        board = chess.Board()
        try:
            best_move = self.engine.search(board, time_limit=1.0)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if best_move and elapsed < 2.0:
                print(f"✅ Performance test passed: {elapsed:.2f}s")
                print(f"   Best move: {best_move}")
            else:
                print(f"⚠️  Performance test concerns: {elapsed:.2f}s")
        
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
        
        print()
    
    def print_test_summary(self):
        """Print test results summary"""
        print("📋 Test Results Summary:")
        print("-" * 30)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! Enhanced nudge system is working correctly.")
        elif passed_tests >= total_tests * 0.7:
            print("⚠️  Most tests passed. Enhanced nudge system is functional.")
        else:
            print("❌ Multiple test failures. Enhanced nudge system needs attention.")
    
    def run_validation(self):
        """Run all validation tests"""
        self.test_database_loading()
        self.test_tactical_awareness()
        self.test_confidence_scoring()
        self.test_move_ordering_enhancement()
        self.test_instant_move_detection()
        self.run_performance_test()
        self.print_test_summary()


def main():
    """Main execution function"""
    validator = EnhancedNudgeValidator()
    validator.run_validation()


if __name__ == "__main__":
    main()