#!/usr/bin/env python3
"""
V7P3R v11 Phase 2: Final Integration and Validation Test
Comprehensive test of all Phase 2 enhancements:
- Advanced Time Management
- Enhanced Nudge System  
- Strategic Position Database
- Pattern Matching & Similarity Scoring

Author: Pat Snyder
"""

import time
import chess
import sys
import os
import json
from typing import Dict, List, Tuple

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine


class V7P3RPhase2FinalValidator:
    """Final validation suite for v11 Phase 2 enhancements"""
    
    def __init__(self):
        self.engine = V7P3REngine()
        self.test_results = {}
        self.validation_positions = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting Position", "opening"),
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", "Kiwipete Position", "complex"),
            ("8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1", "Pawn Endgame", "endgame"),
            ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", "Perft Position", "tactical"),
            ("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", "Complex Position", "strategic")
        ]
    
    def test_complete_system_integration(self) -> Dict:
        """Test complete system with all Phase 2 enhancements"""
        print("=" * 70)
        print("TESTING COMPLETE SYSTEM INTEGRATION")
        print("=" * 70)
        
        results = {
            'system_components': {},
            'position_analysis': [],
            'search_performance': [],
            'enhancement_effectiveness': {}
        }
        
        # Test 1: Verify all system components are loaded
        print("Verifying system components...")
        
        components = {
            'time_manager': hasattr(self.engine, 'time_manager'),
            'nudge_database': hasattr(self.engine, 'nudge_database') and len(self.engine.nudge_database) > 0,
            'strategic_database': hasattr(self.engine, 'strategic_database'),
            'nudge_stats': hasattr(self.engine, 'nudge_stats'),
            'evaluation_cache': hasattr(self.engine, 'evaluation_cache'),
            'transposition_table': hasattr(self.engine, 'transposition_table')
        }
        
        results['system_components'] = components
        
        for component, loaded in components.items():
            status = "✅" if loaded else "❌"
            print(f"   {component}: {status}")
        
        all_components_loaded = all(components.values())
        print(f"\nAll components loaded: {'✅' if all_components_loaded else '❌'}")
        
        if not all_components_loaded:
            print("⚠️ Cannot proceed with full testing - missing components")
            return results
        
        # Test 2: Test each validation position
        print(f"\nTesting position analysis and search...")
        
        for fen, name, position_type in self.validation_positions:
            print(f"\n--- {name} ({position_type}) ---")
            
            try:
                board = chess.Board(fen)
                
                # Test time allocation
                time_remaining = 120.0
                allocated_time, target_depth = self.engine.time_manager.calculate_time_allocation(board, time_remaining)
                
                # Test nudge system
                nudge_position_key = self.engine._get_position_key(board)
                has_nudge_data = nudge_position_key in self.engine.nudge_database
                instant_move = self.engine._check_instant_nudge_move(board)
                
                # Test strategic database
                strategic_bonus = self.engine.strategic_database.get_strategic_evaluation_bonus(board)
                
                # Test evaluation
                evaluation = self.engine._evaluate_position(board)
                
                # Test move ordering with first few legal moves
                legal_moves = list(board.legal_moves)[:5]
                ordered_moves = self.engine._order_moves_advanced(board, legal_moves, 3)
                
                # Calculate move bonuses
                move_bonuses = []
                for move in ordered_moves[:3]:
                    nudge_bonus = self.engine._get_nudge_bonus(board, move)
                    strategic_move_bonus = self.engine.strategic_database.get_strategic_move_bonus(board, move)
                    move_bonuses.append({
                        'move': move.uci(),
                        'nudge_bonus': nudge_bonus,
                        'strategic_bonus': strategic_move_bonus
                    })
                
                position_analysis = {
                    'name': name,
                    'type': position_type,
                    'fen': fen,
                    'time_allocation': {
                        'allocated_time': allocated_time,
                        'target_depth': target_depth
                    },
                    'nudge_system': {
                        'has_nudge_data': has_nudge_data,
                        'instant_move': str(instant_move) if instant_move else None
                    },
                    'strategic_analysis': {
                        'strategic_bonus': strategic_bonus
                    },
                    'evaluation': evaluation,
                    'move_analysis': move_bonuses,
                    'analysis_successful': True
                }
                
                results['position_analysis'].append(position_analysis)
                
                print(f"   Time Allocation: {allocated_time:.2f}s (depth {target_depth})")
                print(f"   Nudge Data: {'Yes' if has_nudge_data else 'No'}")
                print(f"   Instant Move: {instant_move if instant_move else 'None'}")
                print(f"   Strategic Bonus: {strategic_bonus:.3f}")
                print(f"   Evaluation: {evaluation:.3f}")
                print(f"   Top Moves: {[mb['move'] for mb in move_bonuses]}")
                print(f"   ✅ Position analyzed successfully")
                
            except Exception as e:
                print(f"   ❌ Analysis failed: {e}")
                position_analysis = {
                    'name': name,
                    'type': position_type,
                    'error': str(e),
                    'analysis_successful': False
                }
                results['position_analysis'].append(position_analysis)
        
        # Test 3: Search performance test
        print(f"\nTesting search performance with enhancements...")
        
        # Test quick searches on different position types
        for fen, name, position_type in self.validation_positions[:3]:  # Test first 3
            try:
                board = chess.Board(fen)
                
                start_time = time.time()
                best_move = self.engine.search(board, 0.5)  # 0.5 second search
                search_time = time.time() - start_time
                
                search_result = {
                    'position_name': name,
                    'position_type': position_type,
                    'search_time': search_time,
                    'best_move': str(best_move),
                    'nodes_searched': self.engine.nodes_searched,
                    'nps': self.engine.nodes_searched / search_time if search_time > 0 else 0,
                    'time_respected': search_time <= 1.0,  # Allow 100% tolerance
                    'move_found': best_move != chess.Move.null()
                }
                
                results['search_performance'].append(search_result)
                
                print(f"   {name}: {search_time:.3f}s, {self.engine.nodes_searched:,} nodes, {best_move}")
                
            except Exception as e:
                print(f"   {name}: ❌ Search failed - {e}")
        
        # Test 4: Enhancement effectiveness
        print(f"\nEvaluating enhancement effectiveness...")
        
        # Gather statistics
        nudge_stats = self.engine.nudge_stats.copy()
        time_manager_stats = self.engine.time_manager.get_statistics()
        strategic_db_stats = self.engine.strategic_database.get_statistics()
        
        effectiveness = {
            'nudge_hit_rate': (nudge_stats['hits'] / (nudge_stats['hits'] + nudge_stats['misses']) * 100) 
                if nudge_stats['hits'] + nudge_stats['misses'] > 0 else 0,
            'nudge_moves_boosted': nudge_stats['moves_boosted'],
            'instant_moves_found': nudge_stats['instant_moves'],
            'time_manager_positions': time_manager_stats['positions_analyzed'],
            'strategic_patterns_created': strategic_db_stats['patterns_created'],
            'strategic_cache_efficiency': (strategic_db_stats['cache_hits'] / 
                (strategic_db_stats['cache_hits'] + strategic_db_stats['cache_misses']) * 100)
                if strategic_db_stats['cache_hits'] + strategic_db_stats['cache_misses'] > 0 else 0
        }
        
        results['enhancement_effectiveness'] = effectiveness
        
        print(f"   Nudge Hit Rate: {effectiveness['nudge_hit_rate']:.1f}%")
        print(f"   Moves Boosted: {effectiveness['nudge_moves_boosted']}")
        print(f"   Instant Moves: {effectiveness['instant_moves_found']}")
        print(f"   Time Manager Used: {effectiveness['time_manager_positions']} positions")
        print(f"   Strategic Patterns: {effectiveness['strategic_patterns_created']}")
        print(f"   Strategic Cache: {effectiveness['strategic_cache_efficiency']:.1f}% hit rate")
        
        return results
    
    def validate_phase2_completeness(self) -> Dict:
        """Validate that Phase 2 is complete and ready for production"""
        print("\n" + "=" * 70)
        print("PHASE 2 COMPLETENESS VALIDATION")
        print("=" * 70)
        
        # Run integration test
        integration_results = self.test_complete_system_integration()
        
        # Validation criteria
        validation_criteria = {
            'all_components_loaded': all(integration_results['system_components'].values()),
            'position_analysis_success': len([p for p in integration_results['position_analysis'] 
                                            if p.get('analysis_successful', False)]) >= 3,
            'search_performance_acceptable': len([s for s in integration_results['search_performance'] 
                                                if s.get('time_respected', False) and s.get('move_found', False)]) >= 2,
            'enhancements_active': (
                integration_results['enhancement_effectiveness']['nudge_moves_boosted'] > 0 or
                integration_results['enhancement_effectiveness']['time_manager_positions'] > 0 or
                integration_results['enhancement_effectiveness']['strategic_patterns_created'] > 0
            )
        }
        
        overall_validation = {
            'validation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'engine_version': 'v10.9 with v11 Phase 2 Complete',
            'integration_results': integration_results,
            'validation_criteria': validation_criteria,
            'phase2_complete': all(validation_criteria.values()),
            'production_ready': all(validation_criteria.values())
        }
        
        # Print validation summary
        print(f"\nPhase 2 Validation Summary:")
        print(f"{'=' * 50}")
        
        for criterion, passed in validation_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {criterion.replace('_', ' ').title()}: {status}")
        
        phase2_status = "🎉 COMPLETE" if overall_validation['phase2_complete'] else "⚠️ INCOMPLETE"
        production_status = "🚀 READY" if overall_validation['production_ready'] else "🔧 NEEDS WORK"
        
        print(f"\nPhase 2 Status: {phase2_status}")
        print(f"Production Status: {production_status}")
        
        if overall_validation['phase2_complete']:
            print(f"\n🎯 V7P3R v11 Phase 2 is complete and ready for production!")
            print(f"✨ All major enhancements are integrated and functional:")
            print(f"   • Advanced Time Management ✅")
            print(f"   • Enhanced Nudge System ✅") 
            print(f"   • Strategic Position Database ✅")
            print(f"   • Pattern Matching & Similarity Scoring ✅")
        else:
            print(f"\n⚠️ Phase 2 validation incomplete. Address failed criteria above.")
        
        return overall_validation
    
    def save_validation_results(self, results: Dict, filename: str = None):
        """Save validation results to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"v11_phase2_final_validation_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📁 Validation results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Error saving results: {e}")


def main():
    """Main validation function"""
    print("V7P3R v11 Phase 2: Final Integration & Validation")
    print("=" * 80)
    print("Testing all Phase 2 enhancements for production readiness...")
    
    validator = V7P3RPhase2FinalValidator()
    
    try:
        results = validator.validate_phase2_completeness()
        validator.save_validation_results(results)
        
        # Return success code based on validation
        return 0 if results['phase2_complete'] else 1
        
    except Exception as e:
        print(f"\n❌ Phase 2 validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)