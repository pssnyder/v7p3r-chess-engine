#!/usr/bin/env python3
"""
Simple test of refactored metrics system with v7p3r engine only
"""

import sys
import os
import chess
import time

# Add the v7p3r_engine path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'v7p3r_engine'))

from v7p3r_engine.v7p3r import v7p3rEngine
from metrics.refactored_enhanced_metrics_collector import RefactoredEnhancedMetricsCollector
from metrics.enhanced_metrics_store import EnhancedMetricsStore

def test_simple_refactored_metrics():
    """
    Test refactored metrics with a simple engine vs engine scenario
    """
    print("Testing Simple Refactored Metrics Collection")
    print("=" * 50)
    
    try:
        # Initialize components
        engine = v7p3rEngine()
        collector = RefactoredEnhancedMetricsCollector()
        store = EnhancedMetricsStore()
        
        print(f"✓ Engine initialized: {engine.name}")
        print(f"✓ Refactored collector initialized")
        print(f"✓ Enhanced metrics store initialized")
        
        # Test game scenario
        board = chess.Board()
        game_id = "test_refactored_001"
        
        print(f"\nStarting test game: {game_id}")
        print(f"Initial position: {board.fen()}")
        
        # Store game start
        store.start_game(
            game_id=game_id,
            white_player="v7p3r",
            black_player="v7p3r",
            white_config={"name": "v7p3r", "version": "2.0"},
            black_config={"name": "v7p3r", "version": "2.0"},
            pgn_filename=f"{game_id}.pgn"
        )
        
        # Play several moves and collect metrics
        for move_num in range(1, 5):
            print(f"\n--- Move {move_num} ---")
            current_player = "White" if board.turn == chess.WHITE else "Black"
            print(f"Current player: {current_player}")
            
            # Get engine move
            start_time = time.time()
            
            # Ensure board state is set in engine
            engine.search_engine.root_board = board.copy()
            
            # Perform search to populate datasets
            try:
                best_move = engine.search_engine.search(board, board.turn)
                search_time = time.time() - start_time
                
                print(f"Engine selected: {best_move} (time: {search_time:.3f}s)")
                
                # Verify the move is legal
                if best_move in board.legal_moves:
                    # Collect comprehensive metrics before making the move
                    fen_before = board.fen()
                    
                    # Use refactored collector
                    metrics = collector.collect_comprehensive_metrics(
                        engine, board, best_move, time_taken=search_time
                    )
                    
                    print(f"Collected {len(metrics)} metrics components")
                    
                    # Add game-specific information
                    metrics.update({
                        'game_id': game_id,
                        'move_number': move_num,
                        'player_color': 'white' if board.turn == chess.WHITE else 'black',
                        'move_san': board.san(best_move),
                        'move_uci': best_move.uci(),
                        'fen_before': fen_before,
                        'fen_after': '',  # Will be set after move
                        'time_taken': search_time
                    })
                    
                    # Make the move
                    board.push(best_move)
                    metrics['fen_after'] = board.fen()
                    
                    # Store metrics in database
                    try:
                        store.add_enhanced_move_metric(**metrics)
                        print(f"✓ Metrics stored for move {move_num}")
                        
                        # Show key metrics
                        key_info = {
                            'algorithm': metrics.get('search_algorithm', 'N/A'),
                            'depth': metrics.get('depth_reached', 'N/A'),
                            'nodes': metrics.get('nodes_searched', 'N/A'),
                            'score': metrics.get('total_score', 'N/A'),
                            'material': metrics.get('material_balance', 'N/A'),
                            'phase': metrics.get('game_phase', 'N/A')
                        }
                        print(f"Key metrics: {key_info}")
                        
                    except Exception as e:
                        print(f"✗ Failed to store metrics: {e}")
                        
                else:
                    print(f"✗ Illegal move selected: {best_move}")
                    break
                    
            except Exception as e:
                print(f"✗ Search failed: {e}")
                break
        
        print(f"\n--- Final Results ---")
        print(f"Moves played: {len(board.move_stack)}")
        print(f"Final position: {board.fen()}")
        
        # Finish the game in the database
        result = "1/2-1/2"  # Draw for test
        store.finish_game(
            game_id=game_id,
            result=result,
            termination="test_end",
            total_moves=len(board.move_stack),
            game_duration=5.0
        )
        
        print(f"✓ Game finished and recorded as: {result}")
        
        # Query the stored data
        print(f"\n--- Database Verification ---")
        import sqlite3
        conn = sqlite3.connect(store.db_path)
        cursor = conn.cursor()
        
        # Check stored moves for this game
        cursor.execute("""
            SELECT move_number, move_uci, search_algorithm, total_score, material_balance
            FROM move_metrics 
            WHERE game_id = ?
            ORDER BY move_number
        """, (game_id,))
        
        stored_moves = cursor.fetchall()
        print(f"Stored moves in database: {len(stored_moves)}")
        
        for move in stored_moves:
            print(f"  Move {move[0]}: {move[1]} | Algorithm: {move[2]} | Score: {move[3]} | Material: {move[4]}")
        
        conn.close()
        
        print(f"\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_dataset_integration():
    """
    Test the integration between search_dataset and score_dataset
    """
    print(f"\n--- Testing Dataset Integration ---")
    
    try:
        engine = v7p3rEngine()
        collector = RefactoredEnhancedMetricsCollector()
        board = chess.Board()
        
        # Perform search
        print("Performing search to populate datasets...")
        move = engine.search_engine.search(board, chess.WHITE)
        
        # Check search dataset
        if hasattr(engine.search_engine, 'search_dataset'):
            search_data = engine.search_engine.search_dataset
            print(f"Search dataset populated: {len(search_data)} fields")
            
            search_metrics = collector.collect_from_search_dataset(engine.search_engine)
            print(f"Search metrics extracted: {len(search_metrics)} fields")
            
            # Show non-null search metrics
            non_null_search = {k: v for k, v in search_metrics.items() if v not in [None, 0, '', 'unknown']}
            print(f"Non-default search metrics: {len(non_null_search)}")
            for key, value in list(non_null_search.items())[:5]:
                print(f"  {key}: {value}")
        
        # Check score dataset
        if hasattr(engine.scoring_calculator, 'score_dataset'):
            score_data = engine.scoring_calculator.score_dataset
            print(f"\nScore dataset populated: {len(score_data)} fields")
            
            scoring_metrics = collector.collect_from_score_dataset(engine.scoring_calculator)
            print(f"Scoring metrics extracted: {len(scoring_metrics)} fields")
            
            # Show non-zero scoring metrics
            non_zero_scores = {k: v for k, v in scoring_metrics.items() if v != 0.0 and '_score' in k}
            print(f"Non-zero scoring components: {len(non_zero_scores)}")
            for key, value in list(non_zero_scores.items())[:5]:
                print(f"  {key}: {value}")
        
        print(f"✓ Dataset integration test completed")
        
    except Exception as e:
        print(f"Dataset integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_integration()
    test_simple_refactored_metrics()
